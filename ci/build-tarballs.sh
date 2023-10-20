#!/bin/bash

# A small script used for assembling release tarballs for both the `wasmtime`
# binary and the C API. This is executed with two arguments, mostly coming
# from the CI matrix.
#
# * The first argument is the name of the "build", used to name the release.
# * The second argument is the Rust target that the build was performed for.
#
# This expects the build to already be done and will assemble release artifacts
# in `dist/`

set -ex

build=$1
target=$2

rm -rf tmp
mkdir tmp
mkdir -p dist

tag=dev
if [[ $GITHUB_REF == refs/heads/release-* ]]; then
  tag=v$(./ci/print-current-version.sh)
fi

# For *-min builds produce the same named artifacts as the normal build and
# they'll get unioned together in a later step in the CI.
build_pkgname=$build
if [[ $build == *-min ]]; then
  build_pkgname=${build%-min}
fi

bin_pkgname=wasmtime-$tag-$build_pkgname
api_pkgname=wasmtime-$tag-$build_pkgname-c-api

mkdir tmp/$api_pkgname
mkdir tmp/$api_pkgname/lib
mkdir tmp/$api_pkgname/include
mkdir tmp/$bin_pkgname
cp LICENSE README.md tmp/$api_pkgname
cp LICENSE README.md tmp/$bin_pkgname
cp -r crates/c-api/include tmp/$api_pkgname
cp crates/c-api/wasm-c-api/include/wasm.h tmp/$api_pkgname/include

# For *-min builds rename artifacts with a `-min` suffix to avoid eventual
# clashes with the normal builds when the tarballs are unioned together.
if [[ $build == *-min ]]; then
  min="-min"
fi

fmt=tar

case $build in
  x86_64-windows*)
    cp target/$target/release/wasmtime.exe tmp/$bin_pkgname/wasmtime$min.exe
    cp target/$target/release/wasmtime.dll tmp/$api_pkgname/lib/wasmtime$min.dll
    cp target/$target/release/wasmtime.lib tmp/$api_pkgname/lib/wasmtime$min.lib
    cp target/$target/release/wasmtime.dll.lib tmp/$api_pkgname/lib/wasmtime$min.dll.lib
    fmt=zip

    if [ "$min" = "" ]; then
      # Generate a `*.msi` installer for Windows as well
      export WT_VERSION=`cat Cargo.toml | sed -n 's/^version = "\([^"]*\)".*/\1/p'`
      "$WIX/bin/candle" -arch x64 -out target/wasmtime.wixobj ci/wasmtime.wxs
      "$WIX/bin/light" -out dist/$bin_pkgname.msi target/wasmtime.wixobj -ext WixUtilExtension
      rm dist/$bin_pkgname.wixpdb
    fi
    ;;

  x86_64-mingw*)
    cp target/$target/release/wasmtime.exe tmp/$bin_pkgname/wasmtime$min.exe
    cp target/$target/release/wasmtime.dll tmp/$api_pkgname/lib/wasmtime$min.dll
    cp target/$target/release/libwasmtime.a tmp/$api_pkgname/lib/libwasmtime$min.a
    cp target/$target/release/libwasmtime.dll.a tmp/$api_pkgname/lib/libwasmtime$min.dll.a
    fmt=zip
    ;;

  *-macos*)
    # Postprocess the macOS dylib a bit to have a more reasonable `LC_ID_DYLIB`
    # directive than the default one that comes out of the linker when typically
    # doing `cargo build`. For more info see #984
    install_name_tool -id "@rpath/libwasmtime$min.dylib" target/$target/release/libwasmtime.dylib
    cp target/$target/release/wasmtime tmp/$bin_pkgname/wasmtime$min
    cp target/$target/release/libwasmtime.a tmp/$api_pkgname/lib/libwasmtime$min.a
    cp target/$target/release/libwasmtime.dylib tmp/$api_pkgname/lib/libwasmtime$min.dylib
    ;;

  *)
    cp target/$target/release/wasmtime tmp/$bin_pkgname/wasmtime$min
    cp target/$target/release/libwasmtime.a tmp/$api_pkgname/lib/libwasmtime$min.a
    cp target/$target/release/libwasmtime.so tmp/$api_pkgname/lib/libwasmtime$min.so
    ;;
esac


mktarball() {
  dir=$1
  if [ "$fmt" = "tar" ]; then
    tar -czvf dist/$dir.tar.gz -C tmp $dir
  else
    # Note that this runs on Windows, and it looks like GitHub Actions doesn't
    # have a `zip` tool there, so we use something else
    (cd tmp && 7z a ../dist/$dir.zip $dir/)
  fi
}

mktarball $api_pkgname
mktarball $bin_pkgname
