use crate::preview2::filesystem::FileInputStream;
use crate::preview2::{Table, TableError};
use anyhow::Error;
use bytes::Bytes;
use std::fmt;

/// An error which should be reported to Wasm as a runtime error, rather than
/// an error which should trap Wasm execution. The definition for runtime
/// stream errors is the empty type, so the contents of this error will only
/// be available via a `tracing`::event` at `Level::DEBUG`.
pub struct StreamRuntimeError(anyhow::Error);
impl From<anyhow::Error> for StreamRuntimeError {
    fn from(e: anyhow::Error) -> Self {
        StreamRuntimeError(e)
    }
}
impl fmt::Debug for StreamRuntimeError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "Stream runtime error: {:?}", self.0)
    }
}
impl fmt::Display for StreamRuntimeError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "Stream runtime error")
    }
}
impl std::error::Error for StreamRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StreamState {
    Open,
    Closed,
}

impl StreamState {
    pub fn is_closed(&self) -> bool {
        *self == Self::Closed
    }
}

/// Host trait for implementing the `wasi:io/streams.input-stream` resource: A
/// bytestream which can be read from.
#[async_trait::async_trait]
pub trait HostInputStream: Send + Sync {
    /// Read bytes. On success, returns a pair holding the number of bytes
    /// read and a flag indicating whether the end of the stream was reached.
    /// Important: this read must be non-blocking!
    /// Returning an Err which downcasts to a [`StreamRuntimeError`] will be
    /// reported to Wasm as the empty error result. Otherwise, errors will trap.
    fn read(&mut self, size: usize) -> Result<(Bytes, StreamState), Error>;

    /// Read bytes from a stream and discard them. Important: this method must
    /// be non-blocking!
    /// Returning an Error which downcasts to a StreamRuntimeError will be
    /// reported to Wasm as the empty error result. Otherwise, errors will trap.
    fn skip(&mut self, nelem: usize) -> Result<(usize, StreamState), Error> {
        let mut nread = 0;
        let mut state = StreamState::Open;

        let (bs, read_state) = self.read(nelem)?;
        // TODO: handle the case where `bs.len()` is less than `nelem`
        nread += bs.len();
        if read_state.is_closed() {
            state = read_state;
        }

        Ok((nread, state))
    }

    /// Check for read readiness: this method blocks until the stream is ready
    /// for reading.
    /// Returning an error will trap execution.
    async fn ready(&mut self) -> Result<(), Error>;
}

#[derive(Debug)]
pub enum OutputStreamError {
    Closed,
    LastOperationFailed(anyhow::Error),
    Trap(anyhow::Error),
}
impl std::fmt::Display for OutputStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputStreamError::Closed => write!(f, "closed"),
            OutputStreamError::LastOperationFailed(e) => write!(f, "last operation failed: {e}"),
            OutputStreamError::Trap(e) => write!(f, "trap: {e}"),
        }
    }
}
impl std::error::Error for OutputStreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OutputStreamError::Closed => None,
            OutputStreamError::LastOperationFailed(e) | OutputStreamError::Trap(e) => {
                Some(e.as_ref())
            }
        }
    }
}

/// Host trait for implementing the `wasi:io/streams.output-stream` resource:
/// A bytestream which can be written to.
#[async_trait::async_trait]
pub trait HostOutputStream: Send + Sync {
    /// FIXME docs
    fn write(&mut self, bytes: Bytes) -> Result<(), OutputStreamError>;

    /// FIXME docs
    fn flush(&mut self) -> Result<(), OutputStreamError>;

    /// FIXME docs
    async fn write_ready(&mut self) -> Result<usize, OutputStreamError>;

    /// Repeatedly write a byte to a stream. Important: this write must be
    /// non-blocking!
    /// Returning an Err which downcasts to a [`StreamRuntimeError`] will be
    /// reported to Wasm as the empty error result. Otherwise, errors will trap.
    fn write_zeroes(&mut self, nelem: usize) -> Result<(), OutputStreamError> {
        // TODO: We could optimize this to not allocate one big zeroed buffer, and instead write
        // repeatedly from a 'static buffer of zeros.
        let bs = Bytes::from_iter(core::iter::repeat(0 as u8).take(nelem));
        self.write(bs)?;
        Ok(())
    }
}

pub(crate) enum InternalInputStream {
    Host(Box<dyn HostInputStream>),
    File(FileInputStream),
}

pub(crate) trait InternalTableStreamExt {
    fn push_internal_input_stream(
        &mut self,
        istream: InternalInputStream,
    ) -> Result<u32, TableError>;
    fn get_internal_input_stream_mut(
        &mut self,
        fd: u32,
    ) -> Result<&mut InternalInputStream, TableError>;
    fn delete_internal_input_stream(&mut self, fd: u32) -> Result<InternalInputStream, TableError>;
}
impl InternalTableStreamExt for Table {
    fn push_internal_input_stream(
        &mut self,
        istream: InternalInputStream,
    ) -> Result<u32, TableError> {
        self.push(Box::new(istream))
    }
    fn get_internal_input_stream_mut(
        &mut self,
        fd: u32,
    ) -> Result<&mut InternalInputStream, TableError> {
        self.get_mut(fd)
    }
    fn delete_internal_input_stream(&mut self, fd: u32) -> Result<InternalInputStream, TableError> {
        self.delete(fd)
    }
}

/// Extension trait for managing [`HostInputStream`]s and [`HostOutputStream`]s in the [`Table`].
pub trait TableStreamExt {
    /// Push a [`HostInputStream`] into a [`Table`], returning the table index.
    fn push_input_stream(&mut self, istream: Box<dyn HostInputStream>) -> Result<u32, TableError>;
    /// Get a mutable reference to a [`HostInputStream`] in a [`Table`].
    fn get_input_stream_mut(&mut self, fd: u32) -> Result<&mut dyn HostInputStream, TableError>;
    /// Remove [`HostInputStream`] from table:
    fn delete_input_stream(&mut self, fd: u32) -> Result<Box<dyn HostInputStream>, TableError>;

    /// Push a [`HostOutputStream`] into a [`Table`], returning the table index.
    fn push_output_stream(&mut self, ostream: Box<dyn HostOutputStream>)
        -> Result<u32, TableError>;
    /// Get a mutable reference to a [`HostOutputStream`] in a [`Table`].
    fn get_output_stream_mut(&mut self, fd: u32) -> Result<&mut dyn HostOutputStream, TableError>;

    /// Remove [`HostOutputStream`] from table:
    fn delete_output_stream(&mut self, fd: u32) -> Result<Box<dyn HostOutputStream>, TableError>;
}
impl TableStreamExt for Table {
    fn push_input_stream(&mut self, istream: Box<dyn HostInputStream>) -> Result<u32, TableError> {
        self.push_internal_input_stream(InternalInputStream::Host(istream))
    }
    fn get_input_stream_mut(&mut self, fd: u32) -> Result<&mut dyn HostInputStream, TableError> {
        match self.get_internal_input_stream_mut(fd)? {
            InternalInputStream::Host(ref mut h) => Ok(h.as_mut()),
            _ => Err(TableError::WrongType),
        }
    }
    fn delete_input_stream(&mut self, fd: u32) -> Result<Box<dyn HostInputStream>, TableError> {
        let occ = self.entry(fd)?;
        match occ.get().downcast_ref::<InternalInputStream>() {
            Some(InternalInputStream::Host(_)) => {
                let any = occ.remove_entry()?;
                match *any.downcast().expect("downcast checked above") {
                    InternalInputStream::Host(h) => Ok(h),
                    _ => unreachable!("variant checked above"),
                }
            }
            _ => Err(TableError::WrongType),
        }
    }

    fn push_output_stream(
        &mut self,
        ostream: Box<dyn HostOutputStream>,
    ) -> Result<u32, TableError> {
        self.push(Box::new(ostream))
    }
    fn get_output_stream_mut(&mut self, fd: u32) -> Result<&mut dyn HostOutputStream, TableError> {
        let boxed: &mut Box<dyn HostOutputStream> = self.get_mut(fd)?;
        Ok(boxed.as_mut())
    }
    fn delete_output_stream(&mut self, fd: u32) -> Result<Box<dyn HostOutputStream>, TableError> {
        self.delete(fd)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn input_stream_in_table() {
        let dummy = crate::preview2::pipe::ClosedInputStream;
        let mut table = Table::new();
        // Put it into the table:
        let ix = table.push_input_stream(Box::new(dummy)).unwrap();
        // Get a mut ref to it:
        let _ = table.get_input_stream_mut(ix).unwrap();
        // Fails at wrong type:
        assert!(matches!(
            table.get_output_stream_mut(ix),
            Err(TableError::WrongType)
        ));
        // Delete it:
        let _ = table.delete_input_stream(ix).unwrap();
        // Now absent from table:
        assert!(matches!(
            table.get_input_stream_mut(ix),
            Err(TableError::NotPresent)
        ));
    }

    #[test]
    fn output_stream_in_table() {
        let dummy = crate::preview2::pipe::SinkOutputStream;
        let mut table = Table::new();
        // Put it in the table:
        let ix = table.push_output_stream(Box::new(dummy)).unwrap();
        // Get a mut ref to it:
        let _ = table.get_output_stream_mut(ix).unwrap();
        // Fails at wrong type:
        assert!(matches!(
            table.get_input_stream_mut(ix),
            Err(TableError::WrongType)
        ));
        // Delete it:
        let _ = table.delete_output_stream(ix).unwrap();
        // Now absent:
        assert!(matches!(
            table.get_output_stream_mut(ix),
            Err(TableError::NotPresent)
        ));
    }
}
