use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};

use crate::preview2::{
    bindings::{
        sockets::network::{ErrorCode, IpAddressFamily, IpSocketAddress, Network},
        sockets::udp,
    },
    udp::UdpState,
    Table,
};
use crate::preview2::{Pollable, SocketResult, WasiView};
use cap_net_ext::{AddressFamily, PoolExt};
use io_lifetimes::AsSocketlike;
use rustix::io::Errno;
use rustix::net::sockopt;
use wasmtime::component::Resource;

/// Theoretical maximum byte size of a UDP datagram, the real limit is lower,
/// but we do not account for e.g. the transport layer here for simplicity.
/// In practice, datagrams are typically less than 1500 bytes.
const MAX_UDP_DATAGRAM_SIZE: usize = 65535;

fn start_bind(
    table: &mut Table,
    this: Resource<udp::UdpSocket>,
    network: Resource<Network>,
    local_address: IpSocketAddress,
) -> SocketResult<()> {
    let socket = table.get_resource(&this)?;
    match socket.udp_state {
        UdpState::Default => {}
        UdpState::BindStarted | UdpState::Connecting(..) | UdpState::ConnectReady(..) => {
            return Err(ErrorCode::ConcurrencyConflict.into())
        }
        UdpState::Bound | UdpState::Connected(..) => return Err(ErrorCode::AlreadyBound.into()),
    }

    let network = table.get_resource(&network)?;
    let binder = network.pool.udp_binder(local_address)?;

    // Perform the OS bind call.
    binder.bind_existing_udp_socket(
        &*socket
            .udp_socket()
            .as_socketlike_view::<cap_std::net::UdpSocket>(),
    )?;

    let socket = table.get_resource_mut(&this)?;
    socket.udp_state = UdpState::BindStarted;

    Ok(())
}

fn finish_bind(table: &mut Table, this: Resource<udp::UdpSocket>) -> SocketResult<()> {
    let socket = table.get_resource_mut(&this)?;
    match socket.udp_state {
        UdpState::BindStarted => {
            socket.udp_state = UdpState::Bound;
            Ok(())
        }
        _ => Err(ErrorCode::NotInProgress.into()),
    }
}

impl<T: WasiView> udp::Host for T {}

impl<T: WasiView> crate::preview2::host::udp::udp::HostUdpSocket for T {
    fn start_bind(
        &mut self,
        this: Resource<udp::UdpSocket>,
        network: Resource<Network>,
        local_address: IpSocketAddress,
    ) -> SocketResult<()> {
        start_bind(self.table_mut(), this, network, local_address)
    }

    fn finish_bind(&mut self, this: Resource<udp::UdpSocket>) -> SocketResult<()> {
        finish_bind(self.table_mut(), this)
    }

    fn start_connect(
        &mut self,
        this: Resource<udp::UdpSocket>,
        network: Resource<Network>,
        remote_address: IpSocketAddress,
    ) -> SocketResult<()> {
        let table = self.table_mut();
        let socket = table.get_resource(&this)?;
        match socket.udp_state {
            UdpState::Default => {
                let addr = match socket.family {
                    AddressFamily::Ipv4 => Ipv4Addr::UNSPECIFIED.into(),
                    AddressFamily::Ipv6 => Ipv6Addr::UNSPECIFIED.into(),
                };
                start_bind(
                    table,
                    Resource::new_borrow(this.rep()),
                    Resource::new_borrow(network.rep()),
                    SocketAddr::new(addr, 0).into(),
                )?;
                finish_bind(table, Resource::new_borrow(this.rep()))?;
            }
            UdpState::Bound => {}
            UdpState::BindStarted | UdpState::Connecting(..) | UdpState::ConnectReady(..) => {
                return Err(ErrorCode::ConcurrencyConflict.into())
            }
            UdpState::Connected(..) => return Err(ErrorCode::AlreadyConnected.into()),
        }

        let socket = table.get_resource(&this)?;
        let network = table.get_resource(&network)?;
        let connecter = network.pool.udp_connecter(remote_address)?;

        // Do an OS `connect`. Our socket is non-blocking, so it'll either...
        let res = connecter.connect_existing_udp_socket(
            &*socket
                .udp_socket()
                .as_socketlike_view::<cap_std::net::UdpSocket>(),
        );
        match res {
            // succeed immediately,
            Ok(()) => {
                let socket = table.get_resource_mut(&this)?;
                socket.udp_state = UdpState::ConnectReady(remote_address);
                Ok(())
            }
            // continue in progress,
            Err(err) if err.raw_os_error() == Some(INPROGRESS.raw_os_error()) => {
                let socket = table.get_resource_mut(&this)?;
                socket.udp_state = UdpState::Connecting(remote_address);
                Ok(())
            }
            // or fail immediately.
            Err(err) => Err(err.into()),
        }
    }

    fn finish_connect(&mut self, this: Resource<udp::UdpSocket>) -> SocketResult<()> {
        let table = self.table_mut();
        let socket = table.get_resource_mut(&this)?;
        match socket.udp_state {
            UdpState::ConnectReady(addr) => {
                socket.udp_state = UdpState::Connected(addr);
                Ok(())
            }
            UdpState::Connecting(addr) => {
                // Do a `poll` to test for completion, using a timeout of zero
                // to avoid blocking.
                match rustix::event::poll(
                    &mut [rustix::event::PollFd::new(
                        socket.udp_socket(),
                        rustix::event::PollFlags::OUT,
                    )],
                    0,
                ) {
                    Ok(0) => return Err(ErrorCode::WouldBlock.into()),
                    Ok(_) => {}
                    Err(err) => return Err(err.into()),
                }

                // Check whether the connect succeeded.
                match sockopt::get_socket_error(socket.udp_socket()) {
                    Ok(Ok(())) => {
                        socket.udp_state = UdpState::Connected(addr);
                        Ok(())
                    }
                    Err(err) | Ok(Err(err)) => Err(err.into()),
                }
            }
            _ => Err(ErrorCode::NotInProgress.into()),
        }
    }

    fn receive(
        &mut self,
        this: Resource<udp::UdpSocket>,
        max_results: u64,
    ) -> SocketResult<Vec<udp::Datagram>> {
        if max_results == 0 {
            return Ok(vec![]);
        }

        let table = self.table();
        let socket = table.get_resource(&this)?;

        let udp_socket = socket.udp_socket();
        let mut datagrams = vec![];
        let mut buf = [0; MAX_UDP_DATAGRAM_SIZE];
        match socket.udp_state {
            UdpState::Default | UdpState::BindStarted => return Err(ErrorCode::InvalidState.into()),
            UdpState::Bound | UdpState::Connecting(..) | UdpState::ConnectReady(..) => {
                for i in 0..max_results {
                    match udp_socket.try_recv_from(&mut buf) {
                        Ok((size, remote_address)) => datagrams.push(udp::Datagram {
                            data: buf[..size].into(),
                            remote_address: remote_address.into(),
                        }),
                        Err(_e) if i > 0 => {
                            return Ok(datagrams);
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
            }
            UdpState::Connected(remote_address) => {
                for i in 0..max_results {
                    match udp_socket.try_recv(&mut buf) {
                        Ok(size) => datagrams.push(udp::Datagram {
                            data: buf[..size].into(),
                            remote_address,
                        }),
                        Err(_e) if i > 0 => {
                            return Ok(datagrams);
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
            }
        }
        Ok(datagrams)
    }

    fn send(
        &mut self,
        this: Resource<udp::UdpSocket>,
        datagrams: Vec<udp::Datagram>,
    ) -> SocketResult<u64> {
        if datagrams.is_empty() {
            return Ok(0);
        };
        let table = self.table();
        let socket = table.get_resource(&this)?;

        let udp_socket = socket.udp_socket();
        let mut count = 0;
        match socket.udp_state {
            UdpState::Default | UdpState::BindStarted => return Err(ErrorCode::InvalidState.into()),
            UdpState::Bound | UdpState::Connecting(..) | UdpState::ConnectReady(..) => {
                for udp::Datagram {
                    data,
                    remote_address,
                } in datagrams
                {
                    match udp_socket.try_send_to(&data, remote_address.into()) {
                        Ok(_size) => count += 1,
                        Err(_e) if count > 0 => {
                            return Ok(count);
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
            }
            UdpState::Connected(addr) => {
                let addr = SocketAddr::from(addr);
                for udp::Datagram {
                    data,
                    remote_address,
                } in datagrams
                {
                    if SocketAddr::from(remote_address) != addr {
                        // From WIT documentation:
                        // If at least one datagram has been sent successfully, this function never returns an error.
                        if count == 0 {
                            return Err(ErrorCode::InvalidArgument.into());
                        } else {
                            return Ok(count);
                        }
                    }
                    match udp_socket.try_send(&data) {
                        Ok(_size) => count += 1,
                        Err(_e) if count > 0 => {
                            return Ok(count);
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
            }
        }
        Ok(count)
    }

    fn local_address(&mut self, this: Resource<udp::UdpSocket>) -> SocketResult<IpSocketAddress> {
        let table = self.table();
        let socket = table.get_resource(&this)?;
        let addr = socket
            .udp_socket()
            .as_socketlike_view::<std::net::UdpSocket>()
            .local_addr()?;
        Ok(addr.into())
    }

    fn remote_address(&mut self, this: Resource<udp::UdpSocket>) -> SocketResult<IpSocketAddress> {
        let table = self.table();
        let socket = table.get_resource(&this)?;
        let addr = socket
            .udp_socket()
            .as_socketlike_view::<std::net::UdpSocket>()
            .peer_addr()?;
        Ok(addr.into())
    }

    fn address_family(
        &mut self,
        this: Resource<udp::UdpSocket>,
    ) -> Result<IpAddressFamily, anyhow::Error> {
        let table = self.table();
        let socket = table.get_resource(&this)?;
        match socket.family {
            AddressFamily::Ipv4 => Ok(IpAddressFamily::Ipv4),
            AddressFamily::Ipv6 => Ok(IpAddressFamily::Ipv6),
        }
    }

    fn ipv6_only(&mut self, this: Resource<udp::UdpSocket>) -> SocketResult<bool> {
        let table = self.table();
        let socket = table.get_resource(&this)?;
        Ok(sockopt::get_ipv6_v6only(socket.udp_socket())?)
    }

    fn set_ipv6_only(&mut self, this: Resource<udp::UdpSocket>, value: bool) -> SocketResult<()> {
        let table = self.table();
        let socket = table.get_resource(&this)?;
        Ok(sockopt::set_ipv6_v6only(socket.udp_socket(), value)?)
    }

    fn unicast_hop_limit(&mut self, this: Resource<udp::UdpSocket>) -> SocketResult<u8> {
        let table = self.table();
        let socket = table.get_resource(&this)?;

        // We don't track whether the socket is IPv4 or IPv6 so try one and
        // fall back to the other.
        match sockopt::get_ipv6_unicast_hops(socket.udp_socket()) {
            Ok(value) => Ok(value),
            Err(Errno::NOPROTOOPT) => {
                let value = sockopt::get_ip_ttl(socket.udp_socket())?;
                let value = value.try_into().unwrap();
                Ok(value)
            }
            Err(err) => Err(err.into()),
        }
    }

    fn set_unicast_hop_limit(
        &mut self,
        this: Resource<udp::UdpSocket>,
        value: u8,
    ) -> SocketResult<()> {
        let table = self.table();
        let socket = table.get_resource(&this)?;

        // We don't track whether the socket is IPv4 or IPv6 so try one and
        // fall back to the other.
        match sockopt::set_ipv6_unicast_hops(socket.udp_socket(), Some(value)) {
            Ok(()) => Ok(()),
            Err(Errno::NOPROTOOPT) => Ok(sockopt::set_ip_ttl(socket.udp_socket(), value.into())?),
            Err(err) => Err(err.into()),
        }
    }

    fn receive_buffer_size(&mut self, this: Resource<udp::UdpSocket>) -> SocketResult<u64> {
        let table = self.table();
        let socket = table.get_resource(&this)?;
        Ok(sockopt::get_socket_recv_buffer_size(socket.udp_socket())? as u64)
    }

    fn set_receive_buffer_size(
        &mut self,
        this: Resource<udp::UdpSocket>,
        value: u64,
    ) -> SocketResult<()> {
        let table = self.table();
        let socket = table.get_resource(&this)?;
        let value = value.try_into().map_err(|_| ErrorCode::OutOfMemory)?;
        Ok(sockopt::set_socket_recv_buffer_size(
            socket.udp_socket(),
            value,
        )?)
    }

    fn send_buffer_size(&mut self, this: Resource<udp::UdpSocket>) -> SocketResult<u64> {
        let table = self.table();
        let socket = table.get_resource(&this)?;
        Ok(sockopt::get_socket_send_buffer_size(socket.udp_socket())? as u64)
    }

    fn set_send_buffer_size(
        &mut self,
        this: Resource<udp::UdpSocket>,
        value: u64,
    ) -> SocketResult<()> {
        let table = self.table();
        let socket = table.get_resource(&this)?;
        let value = value.try_into().map_err(|_| ErrorCode::OutOfMemory)?;
        Ok(sockopt::set_socket_send_buffer_size(
            socket.udp_socket(),
            value,
        )?)
    }

    fn subscribe(&mut self, this: Resource<udp::UdpSocket>) -> anyhow::Result<Resource<Pollable>> {
        crate::preview2::poll::subscribe(self.table_mut(), this)
    }

    fn drop(&mut self, this: Resource<udp::UdpSocket>) -> Result<(), anyhow::Error> {
        let table = self.table_mut();

        // As in the filesystem implementation, we assume closing a socket
        // doesn't block.
        let dropped = table.delete_resource(this)?;
        drop(dropped);

        Ok(())
    }
}

const INPROGRESS: Errno = if cfg!(windows) {
    // On Windows, non-blocking UDP socket `connect` uses `WSAEWOULDBLOCK`.
    // <https://learn.microsoft.com/en-us/windows/win32/api/winsock2/nf-winsock2-connect>
    Errno::WOULDBLOCK
} else {
    // On POSIX, non-blocking UDP socket `connect` uses `EINPROGRESS`.
    // <https://pubs.opengroup.org/onlinepubs/9699919799/functions/connect.html>
    Errno::INPROGRESS
};
