//! A simple TCP testcase, using IPv4.

use wasi::io::streams;
use wasi::poll::poll;
use wasi::sockets::network::{IpAddressFamily, IpSocketAddress, Ipv4SocketAddress};
use wasi::sockets::{instance_network, network, tcp, tcp_create_socket};
use wasi_sockets_tests::*;

fn wait(sub: poll::Pollable) {
    loop {
        let wait = poll::poll_oneoff(&[sub]);
        if wait[0] {
            break;
        }
    }
}

fn main() {
    let first_message = b"Hello, world!";
    let second_message = b"Greetings, planet!";

    let net = instance_network::instance_network();

    let sock = tcp_create_socket::create_tcp_socket(IpAddressFamily::Ipv4).unwrap();

    tcp::set_listen_backlog_size(sock, 32).unwrap();

    let addr = IpSocketAddress::Ipv4(Ipv4SocketAddress {
        port: 0,                 // use any free port
        address: (127, 0, 0, 1), // localhost
    });

    let sub = tcp::subscribe(sock);

    tcp::start_bind(sock, net, addr).unwrap();
    wait(sub);
    tcp::finish_bind(sock).unwrap();

    tcp::start_listen(sock).unwrap();
    wait(sub);
    tcp::finish_listen(sock).unwrap();

    let addr = tcp::local_address(sock).unwrap();

    let client = tcp_create_socket::create_tcp_socket(IpAddressFamily::Ipv4).unwrap();
    let client_sub = tcp::subscribe(client);

    tcp::start_connect(client, net, addr).unwrap();
    wait(client_sub);
    let (client_input, client_output) = tcp::finish_connect(client).unwrap();

    let (n, status) = streams::write(client_output, &[]).unwrap();
    assert_eq!(n, 0);
    assert_eq!(status, streams::StreamStatus::Open);

    let (n, status) = streams::write(client_output, first_message).unwrap();
    assert_eq!(n, first_message.len() as u64); // Not guaranteed to work but should work in practice.
    assert_eq!(status, streams::StreamStatus::Open);

    streams::drop_input_stream(client_input);
    streams::drop_output_stream(client_output);
    poll::drop_pollable(client_sub);
    tcp::drop_tcp_socket(client);

    wait(sub);
    let (accepted, input, output) = tcp::accept(sock).unwrap();

    let (empty_data, status) = streams::read(input, 0).unwrap();
    assert!(empty_data.is_empty());
    assert_eq!(status, streams::StreamStatus::Open);

    let (data, status) = streams::blocking_read(input, first_message.len() as u64).unwrap();
    assert_eq!(status, streams::StreamStatus::Open);

    tcp::drop_tcp_socket(accepted);
    streams::drop_input_stream(input);
    streams::drop_output_stream(output);

    // Check that we sent and recieved our message!
    assert_eq!(data, first_message); // Not guaranteed to work but should work in practice.

    // Another client
    let client = tcp_create_socket::create_tcp_socket(IpAddressFamily::Ipv4).unwrap();
    let client_sub = tcp::subscribe(client);

    tcp::start_connect(client, net, addr).unwrap();
    wait(client_sub);
    let (client_input, client_output) = tcp::finish_connect(client).unwrap();

    let (n, status) = streams::write(client_output, second_message).unwrap();
    assert_eq!(n, second_message.len() as u64); // Not guaranteed to work but should work in practice.
    assert_eq!(status, streams::StreamStatus::Open);

    streams::drop_input_stream(client_input);
    streams::drop_output_stream(client_output);
    poll::drop_pollable(client_sub);
    tcp::drop_tcp_socket(client);

    wait(sub);
    let (accepted, input, output) = tcp::accept(sock).unwrap();
    let (data, status) = streams::blocking_read(input, second_message.len() as u64).unwrap();
    assert_eq!(status, streams::StreamStatus::Open);

    streams::drop_input_stream(input);
    streams::drop_output_stream(output);
    tcp::drop_tcp_socket(accepted);

    // Check that we sent and recieved our message!
    assert_eq!(data, second_message); // Not guaranteed to work but should work in practice.

    poll::drop_pollable(sub);
    tcp::drop_tcp_socket(sock);
    network::drop_network(net);
}
