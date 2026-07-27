

set -ex



test -f ${PREFIX}/lib/libkeyutils.so
keyctl --version | grep '1.6.3'
exit 0
