use crate::preview2::bindings::{
    io::streams::{InputStream, OutputStream},
    poll::poll::Pollable,
    sockets::network::{self, ErrorCode, IpAddressFamily, IpSocketAddress, Network},
    sockets::tcp::{self, ShutdownType},
};
use crate::preview2::network::TableNetworkExt;
use crate::preview2::poll::TablePollableExt;
use crate::preview2::stream::TableStreamExt;
use crate::preview2::tcp::{HostTcpSocket, HostTcpSocketInner, HostTcpState, TableTcpSocketExt};
use crate::preview2::{HostPollable, PollableFuture, WasiView};
use cap_net_ext::{Blocking, PoolExt, TcpListenerExt};
use io_lifetimes::AsSocketlike;
use rustix::net::sockopt;
use std::any::Any;
use std::mem;
use std::pin::Pin;
use std::sync::Arc;
#[cfg(unix)]
use tokio::task::spawn;
#[cfg(not(unix))]
use tokio::task::spawn_blocking;
use tokio::task::JoinHandle;

impl<T: WasiView> tcp::Host for T {
    fn start_bind(
        &mut self,
        this: tcp::TcpSocket,
        network: Network,
        local_address: IpSocketAddress,
    ) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        let mut tcp_state = socket.inner.tcp_state.write().unwrap();
        match &*tcp_state {
            HostTcpState::Default => {}
            _ => return Err(ErrorCode::NotInProgress.into()),
        }

        let network = table.get_network(network)?;
        let binder = network.0.tcp_binder(local_address)?;

        binder.bind_existing_tcp_listener(socket.tcp_socket())?;

        *tcp_state = HostTcpState::BindStarted;
        socket.inner.sender.send(()).unwrap();

        Ok(())
    }

    // TODO: Bind and listen aren't really blocking operations; figure this
    // out at the spec level.
    fn finish_bind(&mut self, this: tcp::TcpSocket) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        let mut tcp_state = socket.inner.tcp_state.write().unwrap();
        match &mut *tcp_state {
            HostTcpState::BindStarted => {
                *tcp_state = HostTcpState::Bound;
                Ok(())
            }
            _ => Err(ErrorCode::NotInProgress.into()),
        }
    }

    fn start_connect(
        &mut self,
        this: tcp::TcpSocket,
        network: Network,
        remote_address: IpSocketAddress,
    ) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        let mut tcp_state = socket.inner.tcp_state.write().unwrap();
        match &*tcp_state {
            HostTcpState::Default => {}
            HostTcpState::Connected => return Err(ErrorCode::AlreadyConnected.into()),
            _ => return Err(ErrorCode::NotInProgress.into()),
        }

        let network = table.get_network(network)?;
        let connecter = network.0.tcp_connecter(remote_address)?;

        // Do a host `connect`. Our socket is non-blocking, so it'll either...
        match connecter.connect_existing_tcp_listener(socket.tcp_socket()) {
            // succeed immediately,
            Ok(()) => {
                *tcp_state = HostTcpState::ConnectReady(Ok(()));
                return Ok(());
            }
            // continue in progress,
            Err(err)
                if err.raw_os_error() == Some(rustix::io::Errno::INPROGRESS.raw_os_error()) => {}
            // or fail immediately.
            Err(err) => return Err(err.into()),
        }

        // The connect is continuing in progres. Set up the join handle.

        let clone = socket.clone_inner();

        #[cfg(unix)]
        let join = spawn(async move {
            let result = match clone.tcp_socket.writable().await {
                Ok(mut writable) => {
                    writable.retain_ready();

                    // Check whether the connect succeeded.
                    match sockopt::get_socket_error(&clone.tcp_socket) {
                        Ok(Ok(())) => Ok(()),
                        Err(err) | Ok(Err(err)) => Err(err.into()),
                    }
                }
                Err(err) => Err(err),
            };

            *clone.tcp_state.write().unwrap() = HostTcpState::ConnectReady(result);
            clone.sender.send(()).unwrap();
        });

        #[cfg(not(unix))]
        let join = spawn_blocking(move || {
            let result = match rustix::event::poll(
                &mut [rustix::event::PollFd::new(
                    &clone.tcp_socket,
                    rustix::event::PollFlags::OUT,
                )],
                -1,
            ) {
                Ok(_) => {
                    // Check whether the connect succeeded.
                    match sockopt::get_socket_error(&clone.tcp_socket) {
                        Ok(Ok(())) => Ok(()),
                        Err(err) | Ok(Err(err)) => Err(err.into()),
                    }
                }
                Err(err) => Err(err.into()),
            };

            *clone.tcp_state.write().unwrap() = HostTcpState::ConnectReady(result);
            clone.sender.send(()).unwrap();
        });

        *tcp_state = HostTcpState::Connecting(Pin::from(Box::new(join)));

        Ok(())
    }

    fn finish_connect(
        &mut self,
        this: tcp::TcpSocket,
    ) -> Result<(InputStream, OutputStream), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        let mut tcp_state = socket.inner.tcp_state.write().unwrap();
        match &mut *tcp_state {
            HostTcpState::ConnectReady(_) => {}
            HostTcpState::Connecting(join) => match maybe_unwrap_future(join) {
                Some(joined) => joined.unwrap(),
                None => return Err(ErrorCode::WouldBlock.into()),
            },
            _ => return Err(ErrorCode::NotInProgress.into()),
        };

        let old_state = mem::replace(&mut *tcp_state, HostTcpState::Connected);

        // Extract the connection result.
        let result = match old_state {
            HostTcpState::ConnectReady(result) => result,
            _ => panic!(),
        };

        // Report errors, resetting the state if needed.
        match result {
            Ok(()) => {}
            Err(err) => {
                *tcp_state = HostTcpState::Default;
                return Err(err.into());
            }
        }

        drop(tcp_state);

        let input_clone = socket.clone_inner();
        let output_clone = socket.clone_inner();

        let input_stream = self.table_mut().push_input_stream(Box::new(input_clone))?;
        let output_stream = self
            .table_mut()
            .push_output_stream(Box::new(output_clone))?;

        Ok((input_stream, output_stream))
    }

    fn start_listen(
        &mut self,
        this: tcp::TcpSocket,
        _network: Network,
    ) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        let mut tcp_state = socket.inner.tcp_state.write().unwrap();
        match &*tcp_state {
            HostTcpState::Bound => {}
            HostTcpState::ListenStarted => return Err(ErrorCode::AlreadyListening.into()),
            HostTcpState::Connected => return Err(ErrorCode::AlreadyConnected.into()),
            _ => return Err(ErrorCode::NotInProgress.into()),
        }

        socket.tcp_socket().listen(None)?;

        *tcp_state = HostTcpState::ListenStarted;
        socket.inner.sender.send(()).unwrap();

        Ok(())
    }

    fn finish_listen(&mut self, this: tcp::TcpSocket) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        let mut tcp_state = socket.inner.tcp_state.write().unwrap();

        match &mut *tcp_state {
            HostTcpState::ListenStarted => {}
            _ => return Err(ErrorCode::NotInProgress.into()),
        }

        let new_join = spawn_task_to_wait_for_connections(socket.clone_inner());
        *tcp_state = HostTcpState::Listening(Pin::from(Box::new(new_join)));
        drop(tcp_state);

        Ok(())
    }

    fn accept(
        &mut self,
        this: tcp::TcpSocket,
    ) -> Result<(tcp::TcpSocket, InputStream, OutputStream), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        let mut tcp_state = socket.inner.tcp_state.write().unwrap();
        match &mut *tcp_state {
            HostTcpState::ListenReady(_) => {}
            HostTcpState::Listening(join) => match maybe_unwrap_future(join) {
                Some(joined) => joined.unwrap(),
                None => return Err(ErrorCode::WouldBlock.into()),
            },
            HostTcpState::Connected => return Err(ErrorCode::AlreadyConnected.into()),
            _ => return Err(ErrorCode::NotInProgress.into()),
        }

        let new_join = spawn_task_to_wait_for_connections(socket.clone_inner());
        *tcp_state = HostTcpState::Listening(Pin::from(Box::new(new_join)));
        drop(tcp_state);

        // Do the host system call.
        let (connection, _addr) = socket.tcp_socket().accept_with(Blocking::No)?;
        let tcp_socket = HostTcpSocket::from_tcp_stream(connection)?;

        let input_clone = tcp_socket.clone_inner();
        let output_clone = tcp_socket.clone_inner();

        let tcp_socket = self.table_mut().push_tcp_socket(tcp_socket)?;
        let input_stream = self.table_mut().push_input_stream(Box::new(input_clone))?;
        let output_stream = self
            .table_mut()
            .push_output_stream(Box::new(output_clone))?;

        Ok((tcp_socket, input_stream, output_stream))
    }

    fn local_address(&mut self, this: tcp::TcpSocket) -> Result<IpSocketAddress, network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        let addr = socket
            .inner
            .tcp_socket
            .as_socketlike_view::<std::net::TcpStream>()
            .local_addr()?;
        Ok(addr.into())
    }

    fn remote_address(&mut self, this: tcp::TcpSocket) -> Result<IpSocketAddress, network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        let addr = socket
            .inner
            .tcp_socket
            .as_socketlike_view::<std::net::TcpStream>()
            .peer_addr()?;
        Ok(addr.into())
    }

    fn address_family(&mut self, this: tcp::TcpSocket) -> Result<IpAddressFamily, anyhow::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        // If `SO_DOMAIN` is available, use it.
        //
        // TODO: OpenBSD also supports this; upstream PRs are posted.
        #[cfg(not(any(apple, windows, target_os = "netbsd", target_os = "openbsd")))]
        {
            use rustix::net::AddressFamily;

            let family = sockopt::get_socket_domain(socket.tcp_socket())?;
            let family = match family {
                AddressFamily::INET => IpAddressFamily::Ipv4,
                AddressFamily::INET6 => IpAddressFamily::Ipv6,
                _ => return Err(ErrorCode::NotSupported.into()),
            };
            Ok(family)
        }

        // When `SO_DOMAIN` is not available, emulate it.
        #[cfg(any(apple, windows, target_os = "netbsd", target_os = "openbsd"))]
        {
            if let Ok(_) = sockopt::get_ipv6_unicast_hops(socket.tcp_socket()) {
                return Ok(IpAddressFamily::Ipv6);
            }
            if let Ok(_) = sockopt::get_ip_ttl(socket.tcp_socket()) {
                return Ok(IpAddressFamily::Ipv4);
            }
            Err(ErrorCode::NotSupported.into())
        }
    }

    fn ipv6_only(&mut self, this: tcp::TcpSocket) -> Result<bool, network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        Ok(sockopt::get_ipv6_v6only(socket.tcp_socket())?)
    }

    fn set_ipv6_only(&mut self, this: tcp::TcpSocket, value: bool) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        Ok(sockopt::set_ipv6_v6only(socket.tcp_socket(), value)?)
    }

    fn set_listen_backlog_size(
        &mut self,
        this: tcp::TcpSocket,
        value: u64,
    ) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        let tcp_state = socket.inner.tcp_state.read().unwrap();
        match &*tcp_state {
            HostTcpState::Listening(_) => {}
            _ => return Err(ErrorCode::NotInProgress.into()),
        }

        let value = value.try_into().map_err(|_| ErrorCode::OutOfMemory)?;
        Ok(rustix::net::listen(socket.tcp_socket(), value)?)
    }

    fn keep_alive(&mut self, this: tcp::TcpSocket) -> Result<bool, network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        Ok(sockopt::get_socket_keepalive(socket.tcp_socket())?)
    }

    fn set_keep_alive(&mut self, this: tcp::TcpSocket, value: bool) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        Ok(sockopt::set_socket_keepalive(socket.tcp_socket(), value)?)
    }

    fn no_delay(&mut self, this: tcp::TcpSocket) -> Result<bool, network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        Ok(sockopt::get_tcp_nodelay(socket.tcp_socket())?)
    }

    fn set_no_delay(&mut self, this: tcp::TcpSocket, value: bool) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        Ok(sockopt::set_tcp_nodelay(socket.tcp_socket(), value)?)
    }

    fn unicast_hop_limit(&mut self, this: tcp::TcpSocket) -> Result<u8, network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        // We don't track whether the socket is IPv4 or IPv6 so try one and
        // fall back to the other.
        match sockopt::get_ipv6_unicast_hops(socket.tcp_socket()) {
            Ok(value) => Ok(value),
            Err(rustix::io::Errno::NOPROTOOPT) => {
                let value = sockopt::get_ip_ttl(socket.tcp_socket())?;
                let value = value.try_into().unwrap();
                Ok(value)
            }
            Err(err) => Err(err.into()),
        }
    }

    fn set_unicast_hop_limit(
        &mut self,
        this: tcp::TcpSocket,
        value: u8,
    ) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        // We don't track whether the socket is IPv4 or IPv6 so try one and
        // fall back to the other.
        match sockopt::set_ipv6_unicast_hops(socket.tcp_socket(), Some(value)) {
            Ok(()) => Ok(()),
            Err(rustix::io::Errno::NOPROTOOPT) => {
                Ok(sockopt::set_ip_ttl(socket.tcp_socket(), value.into())?)
            }
            Err(err) => Err(err.into()),
        }
    }

    fn receive_buffer_size(&mut self, this: tcp::TcpSocket) -> Result<u64, network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        Ok(sockopt::get_socket_recv_buffer_size(socket.tcp_socket())? as u64)
    }

    fn set_receive_buffer_size(
        &mut self,
        this: tcp::TcpSocket,
        value: u64,
    ) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        let value = value.try_into().map_err(|_| ErrorCode::OutOfMemory)?;
        Ok(sockopt::set_socket_recv_buffer_size(
            socket.tcp_socket(),
            value,
        )?)
    }

    fn send_buffer_size(&mut self, this: tcp::TcpSocket) -> Result<u64, network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        Ok(sockopt::get_socket_send_buffer_size(socket.tcp_socket())? as u64)
    }

    fn set_send_buffer_size(
        &mut self,
        this: tcp::TcpSocket,
        value: u64,
    ) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;
        let value = value.try_into().map_err(|_| ErrorCode::OutOfMemory)?;
        Ok(sockopt::set_socket_send_buffer_size(
            socket.tcp_socket(),
            value,
        )?)
    }

    fn subscribe(&mut self, this: tcp::TcpSocket) -> anyhow::Result<Pollable> {
        fn make_tcp_socket_future<'a>(stream: &'a mut dyn Any) -> PollableFuture<'a> {
            let socket = stream
                .downcast_mut::<HostTcpSocket>()
                .expect("downcast to HostTcpSocket failed");

            Box::pin(async {
                socket.receiver.changed().await.unwrap();
                Ok(())
            })
        }

        let pollable = HostPollable::TableEntry {
            index: this,
            make_future: make_tcp_socket_future,
        };

        Ok(self.table_mut().push_host_pollable(pollable)?)
    }

    fn shutdown(
        &mut self,
        this: tcp::TcpSocket,
        shutdown_type: ShutdownType,
    ) -> Result<(), network::Error> {
        let table = self.table();
        let socket = table.get_tcp_socket(this)?;

        let how = match shutdown_type {
            ShutdownType::Receive => std::net::Shutdown::Read,
            ShutdownType::Send => std::net::Shutdown::Write,
            ShutdownType::Both => std::net::Shutdown::Both,
        };

        socket
            .inner
            .tcp_socket
            .as_socketlike_view::<std::net::TcpStream>()
            .shutdown(how)?;
        Ok(())
    }

    fn drop_tcp_socket(&mut self, this: tcp::TcpSocket) -> Result<(), anyhow::Error> {
        let table = self.table_mut();

        // As in the filesystem implementation, we assume closing a socket
        // doesn't block.
        let dropped = table.delete_tcp_socket(this)?;

        // On non-Unix platforms, do a `shutdown` to wake up `poll`.
        #[cfg(not(unix))]
        rustix::net::shutdown(&dropped.inner.tcp_socket, rustix::net::Shutdown::ReadWrite).unwrap();

        drop(dropped);

        Ok(())
    }
}

/// Spawn a task to monitor a socket for incoming connections that
/// can be `accept`ed.
fn spawn_task_to_wait_for_connections(socket: Arc<HostTcpSocketInner>) -> JoinHandle<()> {
    #[cfg(unix)]
    let new_join = spawn(async move {
        socket.tcp_socket.readable().await.unwrap().retain_ready();
        *socket.tcp_state.write().unwrap() = HostTcpState::ListenReady(Ok(()));
        socket.sender.send(()).unwrap();
    });

    #[cfg(not(unix))]
    let new_join = spawn_blocking(move || {
        let result = match rustix::event::poll(
            &mut [rustix::event::PollFd::new(
                &socket.tcp_socket,
                rustix::event::PollFlags::IN,
            )],
            -1,
        ) {
            Ok(_) => Ok(()),
            Err(err) => Err(err.into()),
        };
        *socket.tcp_state.write().unwrap() = HostTcpState::ListenReady(result);
        socket.sender.send(()).unwrap();
    });

    new_join
}

/// Given a future, return the finished value if it's already ready, or
/// `None` if it's not.
fn maybe_unwrap_future<F: std::future::Future + std::marker::Unpin>(
    future: &mut Pin<Box<F>>,
) -> Option<F::Output> {
    use std::ptr;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    unsafe fn clone(_ptr: *const ()) -> RawWaker {
        const VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    unsafe fn wake(_ptr: *const ()) {}
    unsafe fn wake_by_ref(_ptr: *const ()) {}
    unsafe fn drop(_ptr: *const ()) {}

    let waker = unsafe { Waker::from_raw(clone(ptr::null() as _)) };

    let mut cx = Context::from_waker(&waker);
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(val) => Some(val),
        Poll::Pending => None,
    }
}
