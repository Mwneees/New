use super::{Error, ErrorExt};
use bytes::Bytes;
use std::any::Any;

/// An input bytestream.
///
/// This is "pseudo" because the real streams will be a type in wit, and
/// built into the wit bindings, and will support async and type parameters.
/// This pseudo-stream abstraction is synchronous and only supports bytes.
pub trait InputStream: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    /// Read bytes. On success, returns a pair holding the number of bytes read
    /// and a flag indicating whether the end of the stream was reached.
    fn read(&mut self, _buf: &mut [u8]) -> Result<(u64, bool), Error> {
        Err(Error::badf())
    }

    /// Vectored-I/O form of `read`.
    fn read_vectored<'a>(
        &mut self,
        _bufs: &mut [std::io::IoSliceMut<'a>],
    ) -> Result<(u64, bool), Error> {
        Err(Error::badf())
    }

    /// Test whether vectored I/O reads are known to be optimized in the
    /// underlying implementation.
    fn is_read_vectored(&self) -> bool {
        false
    }

    /// Read bytes from a stream and discard them.
    fn skip(&mut self, nelem: u64) -> Result<(u64, bool), Error> {
        let mut nread = 0;
        let mut saw_end = false;

        // TODO: Optimize by reading more than one byte at a time.
        for _ in 0..nelem {
            let (num, end) = self.read(&mut [0])?;
            nread += num;
            if end {
                saw_end = true;
                break;
            }
        }

        Ok((nread, saw_end))
    }

    /// Return the number of bytes that may be read without blocking.
    fn num_ready_bytes(&self) -> Result<u64, Error> {
        Ok(0)
    }

    /// Test whether this stream is readable.
    fn readable(&self) -> Result<(), Error>;

    /// Test whether this stream is writeable.
    fn writable(&self) -> Result<(), Error>;
}

/// An output bytestream.
///
/// This is "pseudo" because the real streams will be a type in wit, and
/// built into the wit bindings, and will support async and type parameters.
/// This pseudo-stream abstraction is synchronous and only supports bytes.
pub trait OutputStream: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    /// Write bytes. On success, returns the number of bytes written.
    fn write(&mut self, _buf: &[u8]) -> Result<u64, Error> {
        Err(Error::badf())
    }

    /// Vectored-I/O form of `write`.
    fn write_vectored<'a>(&mut self, _bufs: &[std::io::IoSlice<'a>]) -> Result<u64, Error> {
        Err(Error::badf())
    }

    /// Test whether vectored I/O writes are known to be optimized in the
    /// underlying implementation.
    fn is_write_vectored(&self) -> bool {
        false
    }

    /// Read bytes. On success, returns a pair holding the number of bytes read
    /// and a flag indicating whether the end of the stream was reached.
    fn read(&mut self, _buf: &mut [u8]) -> Result<(u64, bool), Error> {
        Err(Error::badf())
    }

    /// Transfer bytes directly from an input stream to an output stream.
    fn splice(&mut self, src: &mut dyn InputStream, nelem: u64) -> Result<(u64, bool), Error> {
        let mut nspliced = 0;
        let mut saw_end = false;

        // TODO: Optimize by splicing more than one byte at a time.
        for _ in 0..nelem {
            let mut buf = [0u8];
            let (num, end) = src.read(&mut buf)?;
            self.write(&buf)?;
            nspliced += num;
            if end {
                saw_end = true;
                break;
            }
        }

        Ok((nspliced, saw_end))
    }

    /// Repeatedly write a byte to a stream.
    fn write_zeroes(&mut self, nelem: u64) -> Result<u64, Error> {
        let mut nwritten = 0;

        // TODO: Optimize by writing more than one byte at a time.
        for _ in 0..nelem {
            let num = self.write(&[0])?;
            if num == 0 {
                break;
            }
            nwritten += num;
        }

        Ok(nwritten)
    }

    /// Test whether this stream is readable.
    fn readable(&self) -> Result<(), Error>;

    /// Test whether this stream is writeable.
    fn writable(&self) -> Result<(), Error>;
}

impl TryInto<Bytes> for &mut Box<dyn OutputStream> {
    type Error = Error;

    fn try_into(self) -> Result<Bytes, Self::Error> {
        self.readable()?;
        let mut buffer = Vec::new();
        let mut eof = false;
        while !eof {
            let buffer_len = 0x400000;
            let mut vec_buffer = vec![0; buffer_len];

            let (bytes_read, end) = self.read(&mut vec_buffer)?;

            let bytes_read = bytes_read as usize;
            vec_buffer.truncate(bytes_read);

            eof = end;
            buffer.append(&mut vec_buffer);
        }
        Ok(Bytes::from(buffer))
    }
}

pub trait TableStreamExt {
    fn get_input_stream(&self, fd: u32) -> Result<&dyn InputStream, Error>;
    fn get_input_stream_mut(&mut self, fd: u32) -> Result<&mut Box<dyn InputStream>, Error>;
    fn delete_input_stream(&mut self, fd: u32) -> Result<(), Error>;

    fn get_output_stream(&self, fd: u32) -> Result<&dyn OutputStream, Error>;
    fn get_output_stream_mut(&mut self, fd: u32) -> Result<&mut Box<dyn OutputStream>, Error>;
    fn delete_output_stream(&mut self, fd: u32) -> Result<(), Error>;
}
impl TableStreamExt for super::Table {
    fn get_input_stream(&self, fd: u32) -> Result<&dyn InputStream, Error> {
        self.get::<Box<dyn InputStream>>(fd).map(|f| f.as_ref())
    }
    fn get_input_stream_mut(&mut self, fd: u32) -> Result<&mut Box<dyn InputStream>, Error> {
        self.get_mut::<Box<dyn InputStream>>(fd)
    }
    fn delete_input_stream(&mut self, fd: u32) -> Result<(), Error> {
        self.delete::<Box<dyn InputStream>>(fd).map(|_old| ())
    }

    fn get_output_stream(&self, fd: u32) -> Result<&dyn OutputStream, Error> {
        self.get::<Box<dyn OutputStream>>(fd).map(|f| f.as_ref())
    }
    fn get_output_stream_mut(&mut self, fd: u32) -> Result<&mut Box<dyn OutputStream>, Error> {
        self.get_mut::<Box<dyn OutputStream>>(fd)
    }
    fn delete_output_stream(&mut self, fd: u32) -> Result<(), Error> {
        self.delete::<Box<dyn OutputStream>>(fd).map(|_old| ())
    }
}
