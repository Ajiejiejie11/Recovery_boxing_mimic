#include <archive.h>

/* Dummy program to test basic compilation and link. */
int main(int argc, const char **argv)
{
    struct archive *a = archive_read_new();
    return archive_read_free(a);
}
