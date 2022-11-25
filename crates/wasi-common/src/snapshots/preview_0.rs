use crate::file::{FileCaps, FileEntryExt, TableFileExt};
use crate::sched::{
    subscription::{RwEventFlags, SubscriptionResult},
    Poll, Userdata,
};
use crate::snapshots::preview_1::types as snapshot1_types;
use crate::snapshots::preview_1::wasi_snapshot_preview1::WasiSnapshotPreview1 as Snapshot1;
use crate::{ErrorExt, WasiCtx};
use cap_std::time::Duration;
use std::collections::HashSet;
use std::convert::{TryFrom, TryInto};
use std::io::{IoSlice, IoSliceMut};
use std::ops::Deref;
use wiggle::GuestPtr;

wiggle::from_witx!({
    witx: ["$WASI_ROOT/phases/old/snapshot_0/witx/wasi_unstable.witx"],
    errors: { errno => trappable Error },
    async: *,
    wasmtime: false,
});

use types::Error;

impl ErrorExt for Error {
    fn not_found() -> Self {
        types::Errno::Noent.into()
    }
    fn too_big() -> Self {
        types::Errno::TooBig.into()
    }
    fn badf() -> Self {
        types::Errno::Badf.into()
    }
    fn exist() -> Self {
        types::Errno::Exist.into()
    }
    fn illegal_byte_sequence() -> Self {
        types::Errno::Ilseq.into()
    }
    fn invalid_argument() -> Self {
        types::Errno::Inval.into()
    }
    fn io() -> Self {
        types::Errno::Io.into()
    }
    fn name_too_long() -> Self {
        types::Errno::Nametoolong.into()
    }
    fn not_dir() -> Self {
        types::Errno::Notdir.into()
    }
    fn not_supported() -> Self {
        types::Errno::Notsup.into()
    }
    fn overflow() -> Self {
        types::Errno::Overflow.into()
    }
    fn range() -> Self {
        types::Errno::Range.into()
    }
    fn seek_pipe() -> Self {
        types::Errno::Spipe.into()
    }
    fn perm() -> Self {
        types::Errno::Perm.into()
    }
}

impl wiggle::GuestErrorType for types::Errno {
    fn success() -> Self {
        Self::Success
    }
}

impl From<wiggle::GuestError> for Error {
    fn from(err: wiggle::GuestError) -> Error {
        snapshot1_types::Error::from(err).into()
    }
}

impl From<snapshot1_types::Error> for Error {
    fn from(error: snapshot1_types::Error) -> Error {
        match error.downcast() {
            Ok(errno) => Error::from(types::Errno::from(errno)),
            Err(trap) => Error::trap(trap),
        }
    }
}

impl From<std::num::TryFromIntError> for Error {
    fn from(_err: std::num::TryFromIntError) -> Error {
        types::Errno::Overflow.into()
    }
}

// Type conversions
// The vast majority of the types defined in `types` and `snapshot1_types` are identical. However,
// since they are defined in separate places for mechanical (wiggle) reasons, we need to manually
// define conversion functions between them.
// Below we have defined these functions as they are needed.

/// Fd is a newtype wrapper around u32. Unwrap and wrap it.
impl From<types::Fd> for snapshot1_types::Fd {
    fn from(fd: types::Fd) -> snapshot1_types::Fd {
        u32::from(fd).into()
    }
}
/// Fd is a newtype wrapper around u32. Unwrap and wrap it.
impl From<snapshot1_types::Fd> for types::Fd {
    fn from(fd: snapshot1_types::Fd) -> types::Fd {
        u32::from(fd).into()
    }
}

/// Trivial conversion between two c-style enums that have the exact same set of variants.
/// Could we do something unsafe and not list all these variants out? Probably, but doing
/// it this way doesn't bother me much. I copy-pasted the list of variants out of the
/// rendered rustdocs.
/// LLVM ought to compile these From impls into no-ops, inshallah
macro_rules! convert_enum {
    ($from:ty, $to:ty, $($var:ident),+) => {
        impl From<$from> for $to {
            fn from(e: $from) -> $to {
                match e {
                    $( <$from>::$var => <$to>::$var, )+
                }
            }
        }
    }
}
convert_enum!(
    snapshot1_types::Errno,
    types::Errno,
    Success,
    TooBig,
    Acces,
    Addrinuse,
    Addrnotavail,
    Afnosupport,
    Again,
    Already,
    Badf,
    Badmsg,
    Busy,
    Canceled,
    Child,
    Connaborted,
    Connrefused,
    Connreset,
    Deadlk,
    Destaddrreq,
    Dom,
    Dquot,
    Exist,
    Fault,
    Fbig,
    Hostunreach,
    Idrm,
    Ilseq,
    Inprogress,
    Intr,
    Inval,
    Io,
    Isconn,
    Isdir,
    Loop,
    Mfile,
    Mlink,
    Msgsize,
    Multihop,
    Nametoolong,
    Netdown,
    Netreset,
    Netunreach,
    Nfile,
    Nobufs,
    Nodev,
    Noent,
    Noexec,
    Nolck,
    Nolink,
    Nomem,
    Nomsg,
    Noprotoopt,
    Nospc,
    Nosys,
    Notconn,
    Notdir,
    Notempty,
    Notrecoverable,
    Notsock,
    Notsup,
    Notty,
    Nxio,
    Overflow,
    Ownerdead,
    Perm,
    Pipe,
    Proto,
    Protonosupport,
    Prototype,
    Range,
    Rofs,
    Spipe,
    Srch,
    Stale,
    Timedout,
    Txtbsy,
    Xdev,
    Notcapable
);
convert_enum!(
    types::Clockid,
    snapshot1_types::Clockid,
    Realtime,
    Monotonic,
    ProcessCputimeId,
    ThreadCputimeId
);

convert_enum!(
    types::Advice,
    snapshot1_types::Advice,
    Normal,
    Sequential,
    Random,
    Willneed,
    Dontneed,
    Noreuse
);
convert_enum!(
    snapshot1_types::Filetype,
    types::Filetype,
    Directory,
    BlockDevice,
    CharacterDevice,
    RegularFile,
    SocketDgram,
    SocketStream,
    SymbolicLink,
    Unknown
);
convert_enum!(types::Whence, snapshot1_types::Whence, Cur, End, Set);

/// Prestat isn't a c-style enum, its a union where the variant has a payload. Its the only one of
/// those we need to convert, so write it by hand.
impl From<snapshot1_types::Prestat> for types::Prestat {
    fn from(p: snapshot1_types::Prestat) -> types::Prestat {
        match p {
            snapshot1_types::Prestat::Dir(d) => types::Prestat::Dir(d.into()),
        }
    }
}

/// Trivial conversion between two structs that have the exact same set of fields,
/// with recursive descent into the field types.
macro_rules! convert_struct {
    ($from:ty, $to:path, $($field:ident),+) => {
        impl From<$from> for $to {
            fn from(e: $from) -> $to {
                $to {
                    $( $field: e.$field.into(), )+
                }
            }
        }
    }
}

convert_struct!(snapshot1_types::PrestatDir, types::PrestatDir, pr_name_len);
convert_struct!(
    snapshot1_types::Fdstat,
    types::Fdstat,
    fs_filetype,
    fs_rights_base,
    fs_rights_inheriting,
    fs_flags
);

/// Snapshot1 Filestat is incompatible with Snapshot0 Filestat - the nlink
/// field is u32 on this Filestat, and u64 on theirs. If you've got more than
/// 2^32 links I don't know what to tell you
impl From<snapshot1_types::Filestat> for types::Filestat {
    fn from(f: snapshot1_types::Filestat) -> types::Filestat {
        types::Filestat {
            dev: f.dev.into(),
            ino: f.ino.into(),
            filetype: f.filetype.into(),
            nlink: f.nlink.try_into().unwrap_or(u32::MAX),
            size: f.size.into(),
            atim: f.atim.into(),
            mtim: f.mtim.into(),
            ctim: f.ctim.into(),
        }
    }
}

/// Trivial conversion between two bitflags that have the exact same set of flags.
macro_rules! convert_flags {
    ($from:ty, $to:ty, $($flag:ident),+) => {
        impl From<$from> for $to {
            fn from(f: $from) -> $to {
                let mut out = <$to>::empty();
                $(
                    if f.contains(<$from>::$flag) {
                        out |= <$to>::$flag;
                    }
                )+
                out
            }
        }
    }
}

/// Need to convert in both directions? This saves listing out the flags twice
macro_rules! convert_flags_bidirectional {
    ($from:ty, $to:ty, $($flag:tt)*) => {
        convert_flags!($from, $to, $($flag)*);
        convert_flags!($to, $from, $($flag)*);
    }
}

convert_flags_bidirectional!(
    snapshot1_types::Fdflags,
    types::Fdflags,
    APPEND,
    DSYNC,
    NONBLOCK,
    RSYNC,
    SYNC
);
convert_flags!(
    types::Lookupflags,
    snapshot1_types::Lookupflags,
    SYMLINK_FOLLOW
);
convert_flags!(
    types::Fstflags,
    snapshot1_types::Fstflags,
    ATIM,
    ATIM_NOW,
    MTIM,
    MTIM_NOW
);
convert_flags!(
    types::Oflags,
    snapshot1_types::Oflags,
    CREAT,
    DIRECTORY,
    EXCL,
    TRUNC
);
convert_flags_bidirectional!(
    types::Rights,
    snapshot1_types::Rights,
    FD_DATASYNC,
    FD_READ,
    FD_SEEK,
    FD_FDSTAT_SET_FLAGS,
    FD_SYNC,
    FD_TELL,
    FD_WRITE,
    FD_ADVISE,
    FD_ALLOCATE,
    PATH_CREATE_DIRECTORY,
    PATH_CREATE_FILE,
    PATH_LINK_SOURCE,
    PATH_LINK_TARGET,
    PATH_OPEN,
    FD_READDIR,
    PATH_READLINK,
    PATH_RENAME_SOURCE,
    PATH_RENAME_TARGET,
    PATH_FILESTAT_GET,
    PATH_FILESTAT_SET_SIZE,
    PATH_FILESTAT_SET_TIMES,
    FD_FILESTAT_GET,
    FD_FILESTAT_SET_SIZE,
    FD_FILESTAT_SET_TIMES,
    PATH_SYMLINK,
    PATH_REMOVE_DIRECTORY,
    PATH_UNLINK_FILE,
    POLL_FD_READWRITE,
    SOCK_SHUTDOWN
);

// This implementation, wherever possible, delegates directly to the Snapshot1 implementation,
// performing the no-op type conversions along the way.
#[wiggle::async_trait]
impl wasi_unstable::WasiUnstable for WasiCtx {
    async fn args_get<'a>(
        &mut self,
        argv: &GuestPtr<'a, GuestPtr<'a, u8>>,
        argv_buf: &GuestPtr<'a, u8>,
    ) -> Result<(), Error> {
        Snapshot1::args_get(self, argv, argv_buf).await?;
        Ok(())
    }

    async fn args_sizes_get(&mut self) -> Result<(types::Size, types::Size), Error> {
        let s = Snapshot1::args_sizes_get(self).await?;
        Ok(s)
    }

    async fn environ_get<'a>(
        &mut self,
        environ: &GuestPtr<'a, GuestPtr<'a, u8>>,
        environ_buf: &GuestPtr<'a, u8>,
    ) -> Result<(), Error> {
        Snapshot1::environ_get(self, environ, environ_buf).await?;
        Ok(())
    }

    async fn environ_sizes_get(&mut self) -> Result<(types::Size, types::Size), Error> {
        let s = Snapshot1::environ_sizes_get(self).await?;
        Ok(s)
    }

    async fn clock_res_get(&mut self, id: types::Clockid) -> Result<types::Timestamp, Error> {
        let t = Snapshot1::clock_res_get(self, id.into()).await?;
        Ok(t)
    }

    async fn clock_time_get(
        &mut self,
        id: types::Clockid,
        precision: types::Timestamp,
    ) -> Result<types::Timestamp, Error> {
        let t = Snapshot1::clock_time_get(self, id.into(), precision).await?;
        Ok(t)
    }

    async fn fd_advise(
        &mut self,
        fd: types::Fd,
        offset: types::Filesize,
        len: types::Filesize,
        advice: types::Advice,
    ) -> Result<(), Error> {
        Snapshot1::fd_advise(self, fd.into(), offset, len, advice.into()).await?;
        Ok(())
    }

    async fn fd_allocate(
        &mut self,
        fd: types::Fd,
        offset: types::Filesize,
        len: types::Filesize,
    ) -> Result<(), Error> {
        Snapshot1::fd_allocate(self, fd.into(), offset, len).await?;
        Ok(())
    }

    async fn fd_close(&mut self, fd: types::Fd) -> Result<(), Error> {
        Snapshot1::fd_close(self, fd.into()).await?;
        Ok(())
    }

    async fn fd_datasync(&mut self, fd: types::Fd) -> Result<(), Error> {
        Snapshot1::fd_datasync(self, fd.into()).await?;
        Ok(())
    }

    async fn fd_fdstat_get(&mut self, fd: types::Fd) -> Result<types::Fdstat, Error> {
        Ok(Snapshot1::fd_fdstat_get(self, fd.into()).await?.into())
    }

    async fn fd_fdstat_set_flags(
        &mut self,
        fd: types::Fd,
        flags: types::Fdflags,
    ) -> Result<(), Error> {
        Snapshot1::fd_fdstat_set_flags(self, fd.into(), flags.into()).await?;
        Ok(())
    }

    async fn fd_fdstat_set_rights(
        &mut self,
        fd: types::Fd,
        fs_rights_base: types::Rights,
        fs_rights_inheriting: types::Rights,
    ) -> Result<(), Error> {
        Snapshot1::fd_fdstat_set_rights(
            self,
            fd.into(),
            fs_rights_base.into(),
            fs_rights_inheriting.into(),
        )
        .await?;
        Ok(())
    }

    async fn fd_filestat_get(&mut self, fd: types::Fd) -> Result<types::Filestat, Error> {
        Ok(Snapshot1::fd_filestat_get(self, fd.into()).await?.into())
    }

    async fn fd_filestat_set_size(
        &mut self,
        fd: types::Fd,
        size: types::Filesize,
    ) -> Result<(), Error> {
        Snapshot1::fd_filestat_set_size(self, fd.into(), size).await?;
        Ok(())
    }

    async fn fd_filestat_set_times(
        &mut self,
        fd: types::Fd,
        atim: types::Timestamp,
        mtim: types::Timestamp,
        fst_flags: types::Fstflags,
    ) -> Result<(), Error> {
        Snapshot1::fd_filestat_set_times(self, fd.into(), atim, mtim, fst_flags.into()).await?;
        Ok(())
    }

    // NOTE on fd_read, fd_pread, fd_write, fd_pwrite implementations:
    // Because the arguments to these function sit behind GuestPtrs, they are not values we
    // can convert and pass to the corresponding function in Snapshot1.
    // Instead, we have copied the implementation of these functions from the Snapshot1 code.
    // The implementations are identical, but the `types::` in scope locally is different.
    // The bodies of these functions is mostly about converting the GuestPtr and types::-based
    // representation to a std::io::IoSlice(Mut) representation.

    async fn fd_read<'a>(
        &mut self,
        fd: types::Fd,
        iovs: &types::IovecArray<'a>,
    ) -> Result<types::Size, Error> {
        let f = self.table().get_file(u32::from(fd))?;
        let f = f.get_cap(FileCaps::READ)?;

        let mut guest_slices: Vec<wiggle::GuestSliceMut<u8>> =
            iovs.iter()
                .map(|iov_ptr| {
                    let iov_ptr = iov_ptr?;
                    let iov: types::Iovec = iov_ptr.read()?;
                    Ok(iov.buf.as_array(iov.buf_len).as_slice_mut()?.expect(
                        "cannot use with shared memories; see https://github.com/bytecodealliance/wasmtime/issues/5235 (TODO)",
                    ))
                })
                .collect::<Result<_, Error>>()?;

        let mut ioslices: Vec<IoSliceMut> = guest_slices
            .iter_mut()
            .map(|s| IoSliceMut::new(&mut *s))
            .collect();

        let bytes_read = f.read_vectored(&mut ioslices).await?;
        Ok(types::Size::try_from(bytes_read)?)
    }

    async fn fd_pread<'a>(
        &mut self,
        fd: types::Fd,
        iovs: &types::IovecArray<'a>,
        offset: types::Filesize,
    ) -> Result<types::Size, Error> {
        let f = self.table().get_file(u32::from(fd))?;
        let f = f.get_cap(FileCaps::READ | FileCaps::SEEK)?;

        let mut guest_slices: Vec<wiggle::GuestSliceMut<u8>> =
            iovs.iter()
                .map(|iov_ptr| {
                    let iov_ptr = iov_ptr?;
                    let iov: types::Iovec = iov_ptr.read()?;
                    Ok(iov.buf.as_array(iov.buf_len).as_slice_mut()?.expect(
                        "cannot use with shared memories; see https://github.com/bytecodealliance/wasmtime/issues/5235 (TODO)",
                    ))
                })
                .collect::<Result<_, Error>>()?;

        let mut ioslices: Vec<IoSliceMut> = guest_slices
            .iter_mut()
            .map(|s| IoSliceMut::new(&mut *s))
            .collect();

        let bytes_read = f.read_vectored_at(&mut ioslices, offset).await?;
        Ok(types::Size::try_from(bytes_read)?)
    }

    async fn fd_write<'a>(
        &mut self,
        fd: types::Fd,
        ciovs: &types::CiovecArray<'a>,
    ) -> Result<types::Size, Error> {
        let f = self.table().get_file(u32::from(fd))?;
        let f = f.get_cap(FileCaps::WRITE)?;

        let guest_slices: Vec<wiggle::GuestSlice<u8>> = ciovs
            .iter()
            .map(|iov_ptr| {
                let iov_ptr = iov_ptr?;
                let iov: types::Ciovec = iov_ptr.read()?;
                Ok(iov
                    .buf
                    .as_array(iov.buf_len)
                    .as_slice()?
                    .expect("cannot use with shared memories; see https://github.com/bytecodealliance/wasmtime/issues/5235 (TODO)"))
            })
            .collect::<Result<_, Error>>()?;

        let ioslices: Vec<IoSlice> = guest_slices
            .iter()
            .map(|s| IoSlice::new(s.deref()))
            .collect();
        let bytes_written = f.write_vectored(&ioslices).await?;

        Ok(types::Size::try_from(bytes_written)?)
    }

    async fn fd_pwrite<'a>(
        &mut self,
        fd: types::Fd,
        ciovs: &types::CiovecArray<'a>,
        offset: types::Filesize,
    ) -> Result<types::Size, Error> {
        let f = self.table().get_file(u32::from(fd))?;
        let f = f.get_cap(FileCaps::WRITE | FileCaps::SEEK)?;

        let guest_slices: Vec<wiggle::GuestSlice<u8>> = ciovs
            .iter()
            .map(|iov_ptr| {
                let iov_ptr = iov_ptr?;
                let iov: types::Ciovec = iov_ptr.read()?;
                Ok(iov
                    .buf
                    .as_array(iov.buf_len)
                    .as_slice()?
                    .expect("cannot use with shared memories; see https://github.com/bytecodealliance/wasmtime/issues/5235 (TODO)"))
            })
            .collect::<Result<_, Error>>()?;

        let ioslices: Vec<IoSlice> = guest_slices
            .iter()
            .map(|s| IoSlice::new(s.deref()))
            .collect();
        let bytes_written = f.write_vectored_at(&ioslices, offset).await?;

        Ok(types::Size::try_from(bytes_written)?)
    }

    async fn fd_prestat_get(&mut self, fd: types::Fd) -> Result<types::Prestat, Error> {
        Ok(Snapshot1::fd_prestat_get(self, fd.into()).await?.into())
    }

    async fn fd_prestat_dir_name<'a>(
        &mut self,
        fd: types::Fd,
        path: &GuestPtr<'a, u8>,
        path_max_len: types::Size,
    ) -> Result<(), Error> {
        Snapshot1::fd_prestat_dir_name(self, fd.into(), path, path_max_len).await?;
        Ok(())
    }

    async fn fd_renumber(&mut self, from: types::Fd, to: types::Fd) -> Result<(), Error> {
        Snapshot1::fd_renumber(self, from.into(), to.into()).await?;
        Ok(())
    }

    async fn fd_seek(
        &mut self,
        fd: types::Fd,
        offset: types::Filedelta,
        whence: types::Whence,
    ) -> Result<types::Filesize, Error> {
        Ok(Snapshot1::fd_seek(self, fd.into(), offset, whence.into()).await?)
    }

    async fn fd_sync(&mut self, fd: types::Fd) -> Result<(), Error> {
        Snapshot1::fd_sync(self, fd.into()).await?;
        Ok(())
    }

    async fn fd_tell(&mut self, fd: types::Fd) -> Result<types::Filesize, Error> {
        Ok(Snapshot1::fd_tell(self, fd.into()).await?)
    }

    async fn fd_readdir<'a>(
        &mut self,
        fd: types::Fd,
        buf: &GuestPtr<'a, u8>,
        buf_len: types::Size,
        cookie: types::Dircookie,
    ) -> Result<types::Size, Error> {
        Ok(Snapshot1::fd_readdir(self, fd.into(), buf, buf_len, cookie).await?)
    }

    async fn path_create_directory<'a>(
        &mut self,
        dirfd: types::Fd,
        path: &GuestPtr<'a, str>,
    ) -> Result<(), Error> {
        Snapshot1::path_create_directory(self, dirfd.into(), path).await?;
        Ok(())
    }

    async fn path_filestat_get<'a>(
        &mut self,
        dirfd: types::Fd,
        flags: types::Lookupflags,
        path: &GuestPtr<'a, str>,
    ) -> Result<types::Filestat, Error> {
        Ok(
            Snapshot1::path_filestat_get(self, dirfd.into(), flags.into(), path)
                .await?
                .into(),
        )
    }

    async fn path_filestat_set_times<'a>(
        &mut self,
        dirfd: types::Fd,
        flags: types::Lookupflags,
        path: &GuestPtr<'a, str>,
        atim: types::Timestamp,
        mtim: types::Timestamp,
        fst_flags: types::Fstflags,
    ) -> Result<(), Error> {
        Snapshot1::path_filestat_set_times(
            self,
            dirfd.into(),
            flags.into(),
            path,
            atim,
            mtim,
            fst_flags.into(),
        )
        .await?;
        Ok(())
    }

    async fn path_link<'a>(
        &mut self,
        src_fd: types::Fd,
        src_flags: types::Lookupflags,
        src_path: &GuestPtr<'a, str>,
        target_fd: types::Fd,
        target_path: &GuestPtr<'a, str>,
    ) -> Result<(), Error> {
        Snapshot1::path_link(
            self,
            src_fd.into(),
            src_flags.into(),
            src_path,
            target_fd.into(),
            target_path,
        )
        .await?;
        Ok(())
    }

    async fn path_open<'a>(
        &mut self,
        dirfd: types::Fd,
        dirflags: types::Lookupflags,
        path: &GuestPtr<'a, str>,
        oflags: types::Oflags,
        fs_rights_base: types::Rights,
        fs_rights_inheriting: types::Rights,
        fdflags: types::Fdflags,
    ) -> Result<types::Fd, Error> {
        Ok(Snapshot1::path_open(
            self,
            dirfd.into(),
            dirflags.into(),
            path,
            oflags.into(),
            fs_rights_base.into(),
            fs_rights_inheriting.into(),
            fdflags.into(),
        )
        .await?
        .into())
    }

    async fn path_readlink<'a>(
        &mut self,
        dirfd: types::Fd,
        path: &GuestPtr<'a, str>,
        buf: &GuestPtr<'a, u8>,
        buf_len: types::Size,
    ) -> Result<types::Size, Error> {
        Ok(Snapshot1::path_readlink(self, dirfd.into(), path, buf, buf_len).await?)
    }

    async fn path_remove_directory<'a>(
        &mut self,
        dirfd: types::Fd,
        path: &GuestPtr<'a, str>,
    ) -> Result<(), Error> {
        Snapshot1::path_remove_directory(self, dirfd.into(), path).await?;
        Ok(())
    }

    async fn path_rename<'a>(
        &mut self,
        src_fd: types::Fd,
        src_path: &GuestPtr<'a, str>,
        dest_fd: types::Fd,
        dest_path: &GuestPtr<'a, str>,
    ) -> Result<(), Error> {
        Snapshot1::path_rename(self, src_fd.into(), src_path, dest_fd.into(), dest_path).await?;
        Ok(())
    }

    async fn path_symlink<'a>(
        &mut self,
        src_path: &GuestPtr<'a, str>,
        dirfd: types::Fd,
        dest_path: &GuestPtr<'a, str>,
    ) -> Result<(), Error> {
        Snapshot1::path_symlink(self, src_path, dirfd.into(), dest_path).await?;
        Ok(())
    }

    async fn path_unlink_file<'a>(
        &mut self,
        dirfd: types::Fd,
        path: &GuestPtr<'a, str>,
    ) -> Result<(), Error> {
        Snapshot1::path_unlink_file(self, dirfd.into(), path).await?;
        Ok(())
    }

    // NOTE on poll_oneoff implementation:
    // Like fd_write and friends, the arguments and return values are behind GuestPtrs,
    // so they are not values we can convert and pass to the poll_oneoff in Snapshot1.
    // Instead, we have copied the implementation of these functions from the Snapshot1 code.
    // The implementations are identical, but the `types::` in scope locally is different.
    // The bodies of these functions is mostly about converting the GuestPtr and types::-based
    // representation to use the Poll abstraction.
    async fn poll_oneoff<'a>(
        &mut self,
        subs: &GuestPtr<'a, types::Subscription>,
        events: &GuestPtr<'a, types::Event>,
        nsubscriptions: types::Size,
    ) -> Result<types::Size, Error> {
        if nsubscriptions == 0 {
            return Err(Error::invalid_argument().context("nsubscriptions must be nonzero"));
        }

        // Special-case a `poll_oneoff` which is just sleeping on a single
        // relative timer event, such as what WASI libc uses to implement sleep
        // functions. This supports all clock IDs, because POSIX says that
        // `clock_settime` doesn't effect relative sleeps.
        if nsubscriptions == 1 {
            let sub = subs.read()?;
            if let types::SubscriptionU::Clock(clocksub) = sub.u {
                if !clocksub
                    .flags
                    .contains(types::Subclockflags::SUBSCRIPTION_CLOCK_ABSTIME)
                {
                    self.sched
                        .sleep(Duration::from_nanos(clocksub.timeout))
                        .await?;
                    events.write(types::Event {
                        userdata: sub.userdata,
                        error: types::Errno::Success,
                        type_: types::Eventtype::Clock,
                        fd_readwrite: fd_readwrite_empty(),
                    })?;
                    return Ok(1);
                }
            }
        }

        let table = &self.table;
        let mut sub_fds: HashSet<types::Fd> = HashSet::new();
        // We need these refmuts to outlive Poll, which will hold the &mut dyn WasiFile inside
        let mut reads: Vec<(u32, Userdata)> = Vec::new();
        let mut writes: Vec<(u32, Userdata)> = Vec::new();
        let mut poll = Poll::new();

        let subs = subs.as_array(nsubscriptions);
        for sub_elem in subs.iter() {
            let sub_ptr = sub_elem?;
            let sub = sub_ptr.read()?;
            match sub.u {
                types::SubscriptionU::Clock(clocksub) => match clocksub.id {
                    types::Clockid::Monotonic => {
                        let clock = self.clocks.monotonic.deref();
                        let precision = Duration::from_nanos(clocksub.precision);
                        let duration = Duration::from_nanos(clocksub.timeout);
                        let deadline = if clocksub
                            .flags
                            .contains(types::Subclockflags::SUBSCRIPTION_CLOCK_ABSTIME)
                        {
                            self.clocks
                                .creation_time
                                .checked_add(duration)
                                .ok_or_else(|| Error::overflow().context("deadline"))?
                        } else {
                            clock
                                .now(precision)
                                .checked_add(duration)
                                .ok_or_else(|| Error::overflow().context("deadline"))?
                        };
                        poll.subscribe_monotonic_clock(
                            clock,
                            deadline,
                            precision,
                            sub.userdata.into(),
                        )
                    }
                    _ => Err(Error::invalid_argument()
                        .context("timer subscriptions only support monotonic timer"))?,
                },
                types::SubscriptionU::FdRead(readsub) => {
                    let fd = readsub.file_descriptor;
                    if sub_fds.contains(&fd) {
                        return Err(Error::invalid_argument()
                            .context("Fd can be subscribed to at most once per poll"));
                    } else {
                        sub_fds.insert(fd);
                    }
                    table
                        .get_file(u32::from(fd))?
                        .get_cap(FileCaps::POLL_READWRITE)?;
                    reads.push((u32::from(fd), sub.userdata.into()));
                }
                types::SubscriptionU::FdWrite(writesub) => {
                    let fd = writesub.file_descriptor;
                    if sub_fds.contains(&fd) {
                        return Err(Error::invalid_argument()
                            .context("Fd can be subscribed to at most once per poll"));
                    } else {
                        sub_fds.insert(fd);
                    }
                    table
                        .get_file(u32::from(fd))?
                        .get_cap(FileCaps::POLL_READWRITE)?;
                    writes.push((u32::from(fd), sub.userdata.into()));
                }
            }
        }

        self.sched.poll_oneoff(&mut poll).await?;

        let results = poll.results();
        let num_results = results.len();
        assert!(
            num_results <= nsubscriptions as usize,
            "results exceeds subscriptions"
        );
        let events = events.as_array(
            num_results
                .try_into()
                .expect("not greater than nsubscriptions"),
        );
        for ((result, userdata), event_elem) in results.into_iter().zip(events.iter()) {
            let event_ptr = event_elem?;
            let userdata: types::Userdata = userdata.into();
            event_ptr.write(match result {
                SubscriptionResult::Read(r) => {
                    let type_ = types::Eventtype::FdRead;
                    match r {
                        Ok((nbytes, flags)) => types::Event {
                            userdata,
                            error: types::Errno::Success,
                            type_,
                            fd_readwrite: types::EventFdReadwrite {
                                nbytes,
                                flags: types::Eventrwflags::from(&flags),
                            },
                        },
                        Err(e) => types::Event {
                            userdata,
                            error: types::Errno::from(e.downcast().map_err(Error::trap)?),
                            type_,
                            fd_readwrite: fd_readwrite_empty(),
                        },
                    }
                }
                SubscriptionResult::Write(r) => {
                    let type_ = types::Eventtype::FdWrite;
                    match r {
                        Ok((nbytes, flags)) => types::Event {
                            userdata,
                            error: types::Errno::Success,
                            type_,
                            fd_readwrite: types::EventFdReadwrite {
                                nbytes,
                                flags: types::Eventrwflags::from(&flags),
                            },
                        },
                        Err(e) => types::Event {
                            userdata,
                            error: types::Errno::from(e.downcast().map_err(Error::trap)?),
                            type_,
                            fd_readwrite: fd_readwrite_empty(),
                        },
                    }
                }
                SubscriptionResult::MonotonicClock(r) => {
                    let type_ = types::Eventtype::Clock;
                    types::Event {
                        userdata,
                        error: match r {
                            Ok(()) => types::Errno::Success,
                            Err(e) => types::Errno::from(e.downcast().map_err(Error::trap)?),
                        },
                        type_,
                        fd_readwrite: fd_readwrite_empty(),
                    }
                }
            })?;
        }

        Ok(num_results.try_into().expect("results fit into memory"))
    }

    async fn proc_exit(&mut self, status: types::Exitcode) -> anyhow::Error {
        Snapshot1::proc_exit(self, status).await
    }

    async fn proc_raise(&mut self, _sig: types::Signal) -> Result<(), Error> {
        Err(Error::trap(anyhow::Error::msg("proc_raise unsupported")))
    }

    async fn sched_yield(&mut self) -> Result<(), Error> {
        Snapshot1::sched_yield(self).await?;
        Ok(())
    }

    async fn random_get<'a>(
        &mut self,
        buf: &GuestPtr<'a, u8>,
        buf_len: types::Size,
    ) -> Result<(), Error> {
        Snapshot1::random_get(self, buf, buf_len).await?;
        Ok(())
    }

    async fn sock_recv<'a>(
        &mut self,
        _fd: types::Fd,
        _ri_data: &types::IovecArray<'a>,
        _ri_flags: types::Riflags,
    ) -> Result<(types::Size, types::Roflags), Error> {
        Err(Error::trap(anyhow::Error::msg("sock_recv unsupported")))
    }

    async fn sock_send<'a>(
        &mut self,
        _fd: types::Fd,
        _si_data: &types::CiovecArray<'a>,
        _si_flags: types::Siflags,
    ) -> Result<types::Size, Error> {
        Err(Error::trap(anyhow::Error::msg("sock_send unsupported")))
    }

    async fn sock_shutdown(&mut self, _fd: types::Fd, _how: types::Sdflags) -> Result<(), Error> {
        Err(Error::trap(anyhow::Error::msg("sock_shutdown unsupported")))
    }
}

impl From<&RwEventFlags> for types::Eventrwflags {
    fn from(flags: &RwEventFlags) -> types::Eventrwflags {
        let mut out = types::Eventrwflags::empty();
        if flags.contains(RwEventFlags::HANGUP) {
            out = out | types::Eventrwflags::FD_READWRITE_HANGUP;
        }
        out
    }
}

fn fd_readwrite_empty() -> types::EventFdReadwrite {
    types::EventFdReadwrite {
        nbytes: 0,
        flags: types::Eventrwflags::empty(),
    }
}
