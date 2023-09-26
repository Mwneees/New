wit_bindgen::generate!("test-command-with-sockets" in "../../wasi/wit");

use wasi::io::streams;
use wasi::poll::poll;
use wasi::sockets::network::{ErrorCode, IpAddressFamily, IpSocketAddress};
use wasi::sockets::{instance_network, tcp, tcp_create_socket};

pub struct PollableResource {
    pub handle: poll::Pollable,
}

impl Drop for PollableResource {
    fn drop(&mut self) {
        poll::drop_pollable(self.handle);
    }
}

impl PollableResource {
    pub fn wait(&self) {
        loop {
            let wait = poll::poll_oneoff(&[self.handle]);
            if wait[0] {
                break;
            }
        }
    }
}

pub struct InputStreamResource {
    pub handle: streams::InputStream,
}

impl Drop for InputStreamResource {
    fn drop(&mut self) {
        streams::drop_input_stream(self.handle);
    }
}

impl InputStreamResource {
    pub fn subscribe(&self) -> PollableResource {
        PollableResource {
            handle: streams::subscribe_to_input_stream(self.handle),
        }
    }
}

pub struct OutputStreamResource {
    pub handle: streams::OutputStream,
}

impl Drop for OutputStreamResource {
    fn drop(&mut self) {
        streams::drop_output_stream(self.handle);
    }
}

impl OutputStreamResource {
    pub fn subscribe(&self) -> PollableResource {
        PollableResource {
            handle: streams::subscribe_to_output_stream(self.handle),
        }
    }

    pub fn write(&self, mut bytes: &[u8]) -> (usize, streams::StreamStatus) {
        let total = bytes.len();
        let mut written = 0;

        let s = self.subscribe();

        while !bytes.is_empty() {
            s.wait();

            let permit = match streams::check_write(self.handle) {
                Ok(n) => n,
                Err(_) => return (written, streams::StreamStatus::Ended),
            };

            let len = bytes.len().min(permit as usize);
            let (chunk, rest) = bytes.split_at(len);

            match streams::write(self.handle, chunk) {
                Ok(()) => {}
                Err(_) => return (written, streams::StreamStatus::Ended),
            }

            match streams::blocking_flush(self.handle) {
                Ok(()) => {}
                Err(_) => return (written, streams::StreamStatus::Ended),
            }

            bytes = rest;
            written += len;
        }

        (total, streams::StreamStatus::Open)
    }
}

pub struct NetworkResource {
    pub handle: wasi::sockets::network::Network,
}

impl Drop for NetworkResource {
    fn drop(&mut self) {
        wasi::sockets::network::drop_network(self.handle);
    }
}

impl NetworkResource {
    pub fn default() -> NetworkResource {
        NetworkResource {
            handle: instance_network::instance_network(),
        }
    }
}

pub struct TcpSocketResource {
    pub handle: wasi::sockets::tcp::TcpSocket,
}

impl Drop for TcpSocketResource {
    fn drop(&mut self) {
        wasi::sockets::tcp::drop_tcp_socket(self.handle);
    }
}

impl TcpSocketResource {
    pub fn new(address_family: IpAddressFamily) -> Result<TcpSocketResource, ErrorCode> {
        Ok(TcpSocketResource {
            handle: tcp_create_socket::create_tcp_socket(address_family)?,
        })
    }

    pub fn subscribe(&self) -> PollableResource {
        PollableResource {
            handle: tcp::subscribe(self.handle),
        }
    }

    pub fn bind(&self, network: &NetworkResource, local_address: IpSocketAddress) -> Result<(), ErrorCode> {
        let sub = self.subscribe();

        tcp::start_bind(self.handle, network.handle, local_address)?;

        loop {
            match tcp::finish_bind(self.handle) {
                Err(ErrorCode::WouldBlock) => sub.wait(),
                result => return result,
            }
        }
    }

    pub fn listen(&self) -> Result<(), ErrorCode> {
        let sub = self.subscribe();

        tcp::start_listen(self.handle)?;

        loop {
            match tcp::finish_listen(self.handle) {
                Err(ErrorCode::WouldBlock) => sub.wait(),
                result => return result,
            }
        }
    }

    pub fn connect(
        &self,
        network: &NetworkResource,
        remote_address: IpSocketAddress,
    ) -> Result<(InputStreamResource, OutputStreamResource), ErrorCode> {
        let sub = self.subscribe();

        tcp::start_connect(self.handle, network.handle, remote_address)?;

        loop {
            match tcp::finish_connect(self.handle) {
                Err(ErrorCode::WouldBlock) => sub.wait(),
                Err(e) => return Err(e),
                Ok((input, output)) => {
                    return Ok((
                        InputStreamResource { handle: input },
                        OutputStreamResource { handle: output },
                    ))
                }
            }
        }
    }

    pub fn accept(
        &self,
    ) -> Result<(TcpSocketResource, InputStreamResource, OutputStreamResource), ErrorCode> {
        let sub = self.subscribe();

        loop {
            match tcp::accept(self.handle) {
                Err(ErrorCode::WouldBlock) => sub.wait(),
                Err(e) => return Err(e),
                Ok((client, input, output)) => {
                    return Ok((
                        TcpSocketResource { handle: client },
                        InputStreamResource { handle: input },
                        OutputStreamResource { handle: output },
                    ))
                }
            }
        }
    }
}
