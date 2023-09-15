use std::any::Any;

use crate::bindings::http::types::{
    Error, Fields, FutureIncomingResponse, FutureTrailers, Headers, IncomingBody, IncomingRequest,
    IncomingResponse, Method, OutgoingBody, OutgoingRequest, OutgoingResponse, ResponseOutparam,
    Scheme, StatusCode, Trailers,
};
use crate::WasiHttpView;
use crate::{
    body::HostIncomingBody,
    types::{
        HeadersRef, HostFields, HostFutureIncomingResponse, HostIncomingResponse,
        HostOutgoingRequest, TableHttpExt,
    },
};
use anyhow::{anyhow, bail, Context};
use wasmtime_wasi::preview2::{
    bindings::io::streams::{InputStream, OutputStream},
    bindings::poll::poll::Pollable,
    pipe::pipe,
    HostPollable, PollableFuture, TablePollableExt, TableStreamExt,
};

#[async_trait::async_trait]
impl<T: WasiHttpView> crate::bindings::http::types::Host for T {
    async fn drop_fields(&mut self, fields: Fields) -> wasmtime::Result<()> {
        self.table()
            .delete_fields(fields)
            .context("[drop_fields] deleting fields")?;
        Ok(())
    }
    async fn new_fields(&mut self, entries: Vec<(String, Vec<u8>)>) -> wasmtime::Result<Fields> {
        let mut map = HostFields::new();
        for (key, value) in entries {
            map.0.insert(key, vec![value.clone()]);
        }

        let id = self
            .table()
            .push_fields(map)
            .context("[new_fields] pushing fields")?;
        Ok(id)
    }
    async fn fields_get(&mut self, fields: Fields, name: String) -> wasmtime::Result<Vec<Vec<u8>>> {
        let res = self
            .table()
            .get_fields(fields)
            .context("[fields_get] getting fields")?
            .0
            .get(&name)
            .ok_or_else(|| anyhow!("key not found: {name}"))?
            .clone();
        Ok(res)
    }
    async fn fields_set(
        &mut self,
        fields: Fields,
        name: String,
        value: Vec<Vec<u8>>,
    ) -> wasmtime::Result<()> {
        let m = self.table().get_fields_mut(fields)?;
        m.0.insert(name, value.clone());
        Ok(())
    }
    async fn fields_delete(&mut self, fields: Fields, name: String) -> wasmtime::Result<()> {
        let m = self.table().get_fields_mut(fields)?;
        m.0.remove(&name);
        Ok(())
    }
    async fn fields_append(
        &mut self,
        fields: Fields,
        name: String,
        value: Vec<u8>,
    ) -> wasmtime::Result<()> {
        let m = self
            .table()
            .get_fields_mut(fields)
            .context("[fields_append] getting mutable fields")?;
        match m.0.get_mut(&name) {
            Some(v) => v.push(value),
            None => {
                let mut vec = std::vec::Vec::new();
                vec.push(value);
                m.0.insert(name, vec);
            }
        };
        Ok(())
    }
    async fn fields_entries(&mut self, fields: Fields) -> wasmtime::Result<Vec<(String, Vec<u8>)>> {
        let result = self
            .table()
            .get_fields(fields)?
            .0
            .iter()
            .map(|(name, value)| (name.clone(), value[0].clone()))
            .collect();
        Ok(result)
    }
    async fn fields_clone(&mut self, fields: Fields) -> wasmtime::Result<Fields> {
        let table = self.table();
        let m = table
            .get_fields(fields)
            .context("[fields_clone] getting fields")?;
        let id = table
            .push_fields(m.clone())
            .context("[fields_clone] pushing fields")?;
        Ok(id)
    }
    async fn drop_incoming_request(&mut self, _request: IncomingRequest) -> wasmtime::Result<()> {
        todo!("we haven't implemented the server side of wasi-http yet")
    }
    async fn drop_outgoing_request(&mut self, request: OutgoingRequest) -> wasmtime::Result<()> {
        self.table().delete_outgoing_request(request)?;
        Ok(())
    }
    async fn incoming_request_method(
        &mut self,
        _request: IncomingRequest,
    ) -> wasmtime::Result<Method> {
        todo!("we haven't implemented the server side of wasi-http yet")
    }
    async fn incoming_request_path_with_query(
        &mut self,
        _request: IncomingRequest,
    ) -> wasmtime::Result<Option<String>> {
        todo!("we haven't implemented the server side of wasi-http yet")
    }
    async fn incoming_request_scheme(
        &mut self,
        _request: IncomingRequest,
    ) -> wasmtime::Result<Option<Scheme>> {
        todo!("we haven't implemented the server side of wasi-http yet")
    }
    async fn incoming_request_authority(
        &mut self,
        _request: IncomingRequest,
    ) -> wasmtime::Result<Option<String>> {
        todo!("we haven't implemented the server side of wasi-http yet")
    }
    async fn incoming_request_headers(
        &mut self,
        _request: IncomingRequest,
    ) -> wasmtime::Result<Headers> {
        todo!("we haven't implemented the server side of wasi-http yet")
    }
    async fn incoming_request_consume(
        &mut self,
        _request: IncomingRequest,
    ) -> wasmtime::Result<Result<InputStream, ()>> {
        todo!("we haven't implemented the server side of wasi-http yet")
    }
    async fn new_outgoing_request(
        &mut self,
        method: Method,
        path_with_query: Option<String>,
        scheme: Option<Scheme>,
        authority: Option<String>,
        headers: Headers,
    ) -> wasmtime::Result<OutgoingRequest> {
        // We're taking ownership of the header data, so remove it from the table.
        let headers = self.table().delete_fields(headers)?;
        let req = HostOutgoingRequest {
            path_with_query: path_with_query.unwrap_or("".to_string()),
            authority: authority.unwrap_or("".to_string()),
            method,
            headers,
            scheme,
            body: None,
        };
        let id = self
            .table()
            .push_outgoing_response(req)
            .context("[new_outgoing_request] pushing request")?;
        Ok(id)
    }
    async fn outgoing_request_write(
        &mut self,
        request: OutgoingRequest,
    ) -> wasmtime::Result<Result<OutputStream, ()>> {
        let req = self
            .table()
            .get_outgoing_request_mut(request)
            .context("[outgoing_request_write] getting request")?;

        if req.body.is_some() {
            return Ok(Err(()));
        }

        let (reader, writer) = pipe(1024 * 1024);

        req.body.replace(reader);

        // The output stream will necessarily outlive the request, because we could be still
        // writing to the stream after `outgoing-handler.handle` is called.
        let writer = self.table().push_output_stream(Box::new(writer))?;

        Ok(Ok(writer))
    }
    async fn drop_response_outparam(
        &mut self,
        _response: ResponseOutparam,
    ) -> wasmtime::Result<()> {
        todo!("we haven't implemented the server side of wasi-http yet")
    }
    async fn set_response_outparam(
        &mut self,
        _outparam: ResponseOutparam,
        _response: Result<OutgoingResponse, Error>,
    ) -> wasmtime::Result<()> {
        todo!("we haven't implemented the server side of wasi-http yet")
    }
    async fn drop_incoming_response(&mut self, response: IncomingResponse) -> wasmtime::Result<()> {
        self.table()
            .delete_incoming_response(response)
            .context("[drop_incoming_response] deleting response")?;
        Ok(())
    }
    async fn drop_outgoing_response(
        &mut self,
        _response: OutgoingResponse,
    ) -> wasmtime::Result<()> {
        bail!("unimplemented: drop_outgoing_response")
    }
    async fn incoming_response_status(
        &mut self,
        response: IncomingResponse,
    ) -> wasmtime::Result<StatusCode> {
        let r = self
            .table()
            .get_incoming_response(response)
            .context("[incoming_response_status] getting response")?;
        Ok(r.status)
    }
    async fn incoming_response_headers(
        &mut self,
        response: IncomingResponse,
    ) -> wasmtime::Result<Headers> {
        let r = self
            .table()
            .get_incoming_response_mut(response)
            .context("[incoming_response_headers] getting response")?;

        let hdrs = match r.headers {
            HeadersRef::Value(ref mut hdrs) => std::mem::take(hdrs),
            HeadersRef::Resource(id) => return Ok(id),
        };

        let id = self.table().push_fields(HostFields::from(hdrs))?;
        self.table().get_incoming_response_mut(response)?.headers = HeadersRef::Resource(id);
        Ok(id)
    }
    async fn incoming_response_consume(
        &mut self,
        response: IncomingResponse,
    ) -> wasmtime::Result<Result<IncomingBody, ()>> {
        let table = self.table();
        let r = table
            .get_incoming_response_mut(response)
            .context("[incoming_response_consume] getting response")?;

        match r.body.take() {
            Some(body) => {
                let id = self
                    .table()
                    .push_incoming_body(HostIncomingBody::new(body))?;
                Ok(Ok(id))
            }

            None => Ok(Err(())),
        }
    }
    async fn drop_future_trailers(&mut self, id: FutureTrailers) -> wasmtime::Result<()> {
        todo!()
    }

    async fn future_trailers_subscribe(
        &mut self,
        id: FutureTrailers,
    ) -> wasmtime::Result<Pollable> {
        let trailers = self.table().get_future_trailers(id)?;
        todo!()
    }
    async fn future_trailers_get(
        &mut self,
        id: FutureTrailers,
    ) -> wasmtime::Result<Option<Result<Trailers, Error>>> {
        let trailers = self.table().get_future_trailers(id)?;
        todo!()
    }
    async fn new_outgoing_response(
        &mut self,
        _status_code: StatusCode,
        _headers: Headers,
    ) -> wasmtime::Result<OutgoingResponse> {
        todo!("we haven't implemented the server side of wasi-http yet")
    }
    async fn outgoing_response_write(
        &mut self,
        _response: OutgoingResponse,
    ) -> wasmtime::Result<Result<OutputStream, ()>> {
        todo!("we haven't implemented the server side of wasi-http yet")
    }
    async fn drop_future_incoming_response(
        &mut self,
        _future: FutureIncomingResponse,
    ) -> wasmtime::Result<()> {
        todo!()
    }
    async fn future_incoming_response_get(
        &mut self,
        id: FutureIncomingResponse,
    ) -> wasmtime::Result<Option<Result<Result<IncomingResponse, Error>, ()>>> {
        if !self.table().get_future_incoming_response(id)?.is_ready() {
            return Ok(None);
        }

        let resp = match self
            .table()
            .delete_future_incoming_response(id)?
            .unwrap_ready()
        {
            Ok(resp) => resp,
            Err(e) => {
                // Trapping if it's not possible to downcast to an wasi-http error
                let e = e.downcast::<Error>()?;
                return Ok(Some(Ok(Err(e))));
            }
        };

        let (parts, body) = resp.resp.into_parts();

        let resp = self.table().push_incoming_response(HostIncomingResponse {
            status: parts.status.as_u16(),
            headers: HeadersRef::Value(parts.headers),
            body: Some(body),
            worker: resp.worker,
        })?;

        Ok(Some(Ok(Ok(resp))))
    }
    async fn listen_to_future_incoming_response(
        &mut self,
        id: FutureIncomingResponse,
    ) -> wasmtime::Result<Pollable> {
        let _ = self.table().get_future_incoming_response(id)?;

        fn make_future<'a>(elem: &'a mut dyn Any) -> PollableFuture<'a> {
            Box::pin(
                elem.downcast_mut::<HostFutureIncomingResponse>()
                    .expect("parent resource is HostFutureIncomingResponse"),
            )
        }

        let pollable = self.table().push_host_pollable(HostPollable::TableEntry {
            index: id,
            make_future,
        })?;

        Ok(pollable)
    }

    async fn incoming_body_stream(
        &mut self,
        id: IncomingBody,
    ) -> wasmtime::Result<Result<InputStream, ()>> {
        let body = self.table().get_incoming_body(id)?;

        if let Some(stream) = body.stream.take() {
            let stream = self.table().push_input_stream_child(Box::new(stream), id)?;
            return Ok(Ok(stream));
        }

        Ok(Err(()))
    }

    async fn incoming_body_finish(&mut self, id: IncomingBody) -> wasmtime::Result<FutureTrailers> {
        let body = self.table().delete_incoming_body(id)?;
        let trailers = self.table().push_future_trailers(body.into_future_trailers())?;
        Ok(trailers)
    }

    async fn drop_incoming_body(&mut self, id: IncomingBody) -> wasmtime::Result<()> {
        let _ = self.table().delete_incoming_body(id)?;
        Ok(())
    }

    async fn outgoing_body_write(
        &mut self,
        _id: OutgoingBody,
    ) -> wasmtime::Result<Result<OutputStream, ()>> {
        todo!()
    }

    async fn outgoing_body_write_trailers(
        &mut self,
        _id: IncomingBody,
        _ts: Trailers,
    ) -> wasmtime::Result<()> {
        todo!()
    }

    async fn drop_outgoing_body(&mut self, _id: OutgoingBody) -> wasmtime::Result<()> {
        todo!()
    }
}
