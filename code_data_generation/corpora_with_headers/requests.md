# === setup.py ===
#!/usr/bin/env python
import os
import sys
from codecs import open

from setuptools import setup

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 8)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write(
        """
==========================
Unsupported Python version
==========================
This version of Requests requires at least Python {}.{}, but
you're trying to install it on Python {}.{}. To resolve this,
consider upgrading to a supported Python version.

If you can't upgrade your Python version, you'll need to
pin to an older version of Requests (<2.32.0).
""".format(
            *(REQUIRED_PYTHON + CURRENT_PYTHON)
        )
    )
    sys.exit(1)


# 'setup.py publish' shortcut.
if sys.argv[-1] == "publish":
    os.system("python setup.py sdist bdist_wheel")
    os.system("twine upload dist/*")
    sys.exit()

requires = [
    "charset_normalizer>=2,<4",
    "idna>=2.5,<4",
    "urllib3>=1.21.1,<3",
    "certifi>=2017.4.17",
]
test_requirements = [
    "pytest-httpbin==2.1.0",
    "pytest-cov",
    "pytest-mock",
    "pytest-xdist",
    "PySocks>=1.5.6, !=1.5.7",
    "pytest>=3",
]

about = {}
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "src", "requests", "__version__.py"), "r", "utf-8") as f:
    exec(f.read(), about)

with open("README.md", "r", "utf-8") as f:
    readme = f.read()

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    long_description=readme,
    long_description_content_type="text/markdown",
    author=about["__author__"],
    author_email=about["__author_email__"],
    url=about["__url__"],
    packages=["requests"],
    package_data={"": ["LICENSE", "NOTICE"]},
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requires,
    license=about["__license__"],
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Software Development :: Libraries",
    ],
    tests_require=test_requirements,
    extras_require={
        "security": [],
        "socks": ["PySocks>=1.5.6, !=1.5.7"],
        "use_chardet_on_py3": ["chardet>=3.0.2,<6"],
    },
    project_urls={
        "Documentation": "https://requests.readthedocs.io",
        "Source": "https://github.com/psf/requests",
    },
)


# === tests/test_utils.py ===
import copy
import filecmp
import os
import tarfile
import zipfile
from collections import deque
from io import BytesIO
from unittest import mock

import pytest

from requests import compat
from requests._internal_utils import unicode_is_ascii
from requests.cookies import RequestsCookieJar
from requests.structures import CaseInsensitiveDict
from requests.utils import (
    _parse_content_type_header,
    add_dict_to_cookiejar,
    address_in_network,
    dotted_netmask,
    extract_zipped_paths,
    get_auth_from_url,
    get_encoding_from_headers,
    get_encodings_from_content,
    get_environ_proxies,
    guess_filename,
    guess_json_utf,
    is_ipv4_address,
    is_valid_cidr,
    iter_slices,
    parse_dict_header,
    parse_header_links,
    prepend_scheme_if_needed,
    requote_uri,
    select_proxy,
    set_environ,
    should_bypass_proxies,
    super_len,
    to_key_val_list,
    to_native_string,
    unquote_header_value,
    unquote_unreserved,
    urldefragauth,
)

from .compat import StringIO, cStringIO


class TestSuperLen:
    @pytest.mark.parametrize(
        "stream, value",
        (
            (StringIO.StringIO, "Test"),
            (BytesIO, b"Test"),
            pytest.param(
                cStringIO, "Test", marks=pytest.mark.skipif("cStringIO is None")
            ),
        ),
    )
    def test_io_streams(self, stream, value):
        """Ensures that we properly deal with different kinds of IO streams."""
        assert super_len(stream()) == 0
        assert super_len(stream(value)) == 4

    def test_super_len_correctly_calculates_len_of_partially_read_file(self):
        """Ensure that we handle partially consumed file like objects."""
        s = StringIO.StringIO()
        s.write("foobarbogus")
        assert super_len(s) == 0

    @pytest.mark.parametrize("error", [IOError, OSError])
    def test_super_len_handles_files_raising_weird_errors_in_tell(self, error):
        """If tell() raises errors, assume the cursor is at position zero."""

        class BoomFile:
            def __len__(self):
                return 5

            def tell(self):
                raise error()

        assert super_len(BoomFile()) == 0

    @pytest.mark.parametrize("error", [IOError, OSError])
    def test_super_len_tell_ioerror(self, error):
        """Ensure that if tell gives an IOError super_len doesn't fail"""

        class NoLenBoomFile:
            def tell(self):
                raise error()

            def seek(self, offset, whence):
                pass

        assert super_len(NoLenBoomFile()) == 0

    def test_string(self):
        assert super_len("Test") == 4

    @pytest.mark.parametrize(
        "mode, warnings_num",
        (
            ("r", 1),
            ("rb", 0),
        ),
    )
    def test_file(self, tmpdir, mode, warnings_num, recwarn):
        file_obj = tmpdir.join("test.txt")
        file_obj.write("Test")
        with file_obj.open(mode) as fd:
            assert super_len(fd) == 4
        assert len(recwarn) == warnings_num

    def test_tarfile_member(self, tmpdir):
        file_obj = tmpdir.join("test.txt")
        file_obj.write("Test")

        tar_obj = str(tmpdir.join("test.tar"))
        with tarfile.open(tar_obj, "w") as tar:
            tar.add(str(file_obj), arcname="test.txt")

        with tarfile.open(tar_obj) as tar:
            member = tar.extractfile("test.txt")
            assert super_len(member) == 4

    def test_super_len_with__len__(self):
        foo = [1, 2, 3, 4]
        len_foo = super_len(foo)
        assert len_foo == 4

    def test_super_len_with_no__len__(self):
        class LenFile:
            def __init__(self):
                self.len = 5

        assert super_len(LenFile()) == 5

    def test_super_len_with_tell(self):
        foo = StringIO.StringIO("12345")
        assert super_len(foo) == 5
        foo.read(2)
        assert super_len(foo) == 3

    def test_super_len_with_fileno(self):
        with open(__file__, "rb") as f:
            length = super_len(f)
            file_data = f.read()
        assert length == len(file_data)

    def test_super_len_with_no_matches(self):
        """Ensure that objects without any length methods default to 0"""
        assert super_len(object()) == 0


class TestToKeyValList:
    @pytest.mark.parametrize(
        "value, expected",
        (
            ([("key", "val")], [("key", "val")]),
            ((("key", "val"),), [("key", "val")]),
            ({"key": "val"}, [("key", "val")]),
            (None, None),
        ),
    )
    def test_valid(self, value, expected):
        assert to_key_val_list(value) == expected

    def test_invalid(self):
        with pytest.raises(ValueError):
            to_key_val_list("string")


class TestUnquoteHeaderValue:
    @pytest.mark.parametrize(
        "value, expected",
        (
            (None, None),
            ("Test", "Test"),
            ('"Test"', "Test"),
            ('"Test\\\\"', "Test\\"),
            ('"\\\\Comp\\Res"', "\\Comp\\Res"),
        ),
    )
    def test_valid(self, value, expected):
        assert unquote_header_value(value) == expected

    def test_is_filename(self):
        assert unquote_header_value('"\\\\Comp\\Res"', True) == "\\\\Comp\\Res"


class TestGetEnvironProxies:
    """Ensures that IP addresses are correctly matches with ranges
    in no_proxy variable.
    """

    @pytest.fixture(autouse=True, params=["no_proxy", "NO_PROXY"])
    def no_proxy(self, request, monkeypatch):
        monkeypatch.setenv(
            request.param, "192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1"
        )

    @pytest.mark.parametrize(
        "url",
        (
            "http://192.168.0.1:5000/",
            "http://192.168.0.1/",
            "http://172.16.1.1/",
            "http://172.16.1.1:5000/",
            "http://localhost.localdomain:5000/v1.0/",
        ),
    )
    def test_bypass(self, url):
        assert get_environ_proxies(url, no_proxy=None) == {}

    @pytest.mark.parametrize(
        "url",
        (
            "http://192.168.1.1:5000/",
            "http://192.168.1.1/",
            "http://www.requests.com/",
        ),
    )
    def test_not_bypass(self, url):
        assert get_environ_proxies(url, no_proxy=None) != {}

    @pytest.mark.parametrize(
        "url",
        (
            "http://192.168.1.1:5000/",
            "http://192.168.1.1/",
            "http://www.requests.com/",
        ),
    )
    def test_bypass_no_proxy_keyword(self, url):
        no_proxy = "192.168.1.1,requests.com"
        assert get_environ_proxies(url, no_proxy=no_proxy) == {}

    @pytest.mark.parametrize(
        "url",
        (
            "http://192.168.0.1:5000/",
            "http://192.168.0.1/",
            "http://172.16.1.1/",
            "http://172.16.1.1:5000/",
            "http://localhost.localdomain:5000/v1.0/",
        ),
    )
    def test_not_bypass_no_proxy_keyword(self, url, monkeypatch):
        # This is testing that the 'no_proxy' argument overrides the
        # environment variable 'no_proxy'
        monkeypatch.setenv("http_proxy", "http://proxy.example.com:3128/")
        no_proxy = "192.168.1.1,requests.com"
        assert get_environ_proxies(url, no_proxy=no_proxy) != {}


class TestIsIPv4Address:
    def test_valid(self):
        assert is_ipv4_address("8.8.8.8")

    @pytest.mark.parametrize("value", ("8.8.8.8.8", "localhost.localdomain"))
    def test_invalid(self, value):
        assert not is_ipv4_address(value)


class TestIsValidCIDR:
    def test_valid(self):
        assert is_valid_cidr("192.168.1.0/24")

    @pytest.mark.parametrize(
        "value",
        (
            "8.8.8.8",
            "192.168.1.0/a",
            "192.168.1.0/128",
            "192.168.1.0/-1",
            "192.168.1.999/24",
        ),
    )
    def test_invalid(self, value):
        assert not is_valid_cidr(value)


class TestAddressInNetwork:
    def test_valid(self):
        assert address_in_network("192.168.1.1", "192.168.1.0/24")

    def test_invalid(self):
        assert not address_in_network("172.16.0.1", "192.168.1.0/24")


class TestGuessFilename:
    @pytest.mark.parametrize(
        "value",
        (1, type("Fake", (object,), {"name": 1})()),
    )
    def test_guess_filename_invalid(self, value):
        assert guess_filename(value) is None

    @pytest.mark.parametrize(
        "value, expected_type",
        (
            (b"value", compat.bytes),
            (b"value".decode("utf-8"), compat.str),
        ),
    )
    def test_guess_filename_valid(self, value, expected_type):
        obj = type("Fake", (object,), {"name": value})()
        result = guess_filename(obj)
        assert result == value
        assert isinstance(result, expected_type)


class TestExtractZippedPaths:
    @pytest.mark.parametrize(
        "path",
        (
            "/",
            __file__,
            pytest.__file__,
            "/etc/invalid/location",
        ),
    )
    def test_unzipped_paths_unchanged(self, path):
        assert path == extract_zipped_paths(path)

    def test_zipped_paths_extracted(self, tmpdir):
        zipped_py = tmpdir.join("test.zip")
        with zipfile.ZipFile(zipped_py.strpath, "w") as f:
            f.write(__file__)

        _, name = os.path.splitdrive(__file__)
        zipped_path = os.path.join(zipped_py.strpath, name.lstrip(r"\/"))
        extracted_path = extract_zipped_paths(zipped_path)

        assert extracted_path != zipped_path
        assert os.path.exists(extracted_path)
        assert filecmp.cmp(extracted_path, __file__)

    def test_invalid_unc_path(self):
        path = r"\\localhost\invalid\location"
        assert extract_zipped_paths(path) == path


class TestContentEncodingDetection:
    def test_none(self):
        encodings = get_encodings_from_content("")
        assert not len(encodings)

    @pytest.mark.parametrize(
        "content",
        (
            # HTML5 meta charset attribute
            '<meta charset="UTF-8">',
            # HTML4 pragma directive
            '<meta http-equiv="Content-type" content="text/html;charset=UTF-8">',
            # XHTML 1.x served with text/html MIME type
            '<meta http-equiv="Content-type" content="text/html;charset=UTF-8" />',
            # XHTML 1.x served as XML
            '<?xml version="1.0" encoding="UTF-8"?>',
        ),
    )
    def test_pragmas(self, content):
        encodings = get_encodings_from_content(content)
        assert len(encodings) == 1
        assert encodings[0] == "UTF-8"

    def test_precedence(self):
        content = """
        <?xml version="1.0" encoding="XML"?>
        <meta charset="HTML5">
        <meta http-equiv="Content-type" content="text/html;charset=HTML4" />
        """.strip()
        assert get_encodings_from_content(content) == ["HTML5", "HTML4", "XML"]


class TestGuessJSONUTF:
    @pytest.mark.parametrize(
        "encoding",
        (
            "utf-32",
            "utf-8-sig",
            "utf-16",
            "utf-8",
            "utf-16-be",
            "utf-16-le",
            "utf-32-be",
            "utf-32-le",
        ),
    )
    def test_encoded(self, encoding):
        data = "{}".encode(encoding)
        assert guess_json_utf(data) == encoding

    def test_bad_utf_like_encoding(self):
        assert guess_json_utf(b"\x00\x00\x00\x00") is None

    @pytest.mark.parametrize(
        ("encoding", "expected"),
        (
            ("utf-16-be", "utf-16"),
            ("utf-16-le", "utf-16"),
            ("utf-32-be", "utf-32"),
            ("utf-32-le", "utf-32"),
        ),
    )
    def test_guess_by_bom(self, encoding, expected):
        data = "\ufeff{}".encode(encoding)
        assert guess_json_utf(data) == expected


USER = PASSWORD = "%!*'();:@&=+$,/?#[] "
ENCODED_USER = compat.quote(USER, "")
ENCODED_PASSWORD = compat.quote(PASSWORD, "")


@pytest.mark.parametrize(
    "url, auth",
    (
        (
            f"http://{ENCODED_USER}:{ENCODED_PASSWORD}@request.com/url.html#test",
            (USER, PASSWORD),
        ),
        ("http://user:pass@complex.url.com/path?query=yes", ("user", "pass")),
        (
            "http://user:pass%20pass@complex.url.com/path?query=yes",
            ("user", "pass pass"),
        ),
        ("http://user:pass pass@complex.url.com/path?query=yes", ("user", "pass pass")),
        (
            "http://user%25user:pass@complex.url.com/path?query=yes",
            ("user%user", "pass"),
        ),
        (
            "http://user:pass%23pass@complex.url.com/path?query=yes",
            ("user", "pass#pass"),
        ),
        ("http://complex.url.com/path?query=yes", ("", "")),
    ),
)
def test_get_auth_from_url(url, auth):
    assert get_auth_from_url(url) == auth


@pytest.mark.parametrize(
    "uri, expected",
    (
        (
            # Ensure requoting doesn't break expectations
            "http://example.com/fiz?buz=%25ppicture",
            "http://example.com/fiz?buz=%25ppicture",
        ),
        (
            # Ensure we handle unquoted percent signs in redirects
            "http://example.com/fiz?buz=%ppicture",
            "http://example.com/fiz?buz=%25ppicture",
        ),
    ),
)
def test_requote_uri_with_unquoted_percents(uri, expected):
    """See: https://github.com/psf/requests/issues/2356"""
    assert requote_uri(uri) == expected


@pytest.mark.parametrize(
    "uri, expected",
    (
        (
            # Illegal bytes
            "http://example.com/?a=%--",
            "http://example.com/?a=%--",
        ),
        (
            # Reserved characters
            "http://example.com/?a=%300",
            "http://example.com/?a=00",
        ),
    ),
)
def test_unquote_unreserved(uri, expected):
    assert unquote_unreserved(uri) == expected


@pytest.mark.parametrize(
    "mask, expected",
    (
        (8, "255.0.0.0"),
        (24, "255.255.255.0"),
        (25, "255.255.255.128"),
    ),
)
def test_dotted_netmask(mask, expected):
    assert dotted_netmask(mask) == expected


http_proxies = {
    "http": "http://http.proxy",
    "http://some.host": "http://some.host.proxy",
}
all_proxies = {
    "all": "socks5://http.proxy",
    "all://some.host": "socks5://some.host.proxy",
}
mixed_proxies = {
    "http": "http://http.proxy",
    "http://some.host": "http://some.host.proxy",
    "all": "socks5://http.proxy",
}


@pytest.mark.parametrize(
    "url, expected, proxies",
    (
        ("hTTp://u:p@Some.Host/path", "http://some.host.proxy", http_proxies),
        ("hTTp://u:p@Other.Host/path", "http://http.proxy", http_proxies),
        ("hTTp:///path", "http://http.proxy", http_proxies),
        ("hTTps://Other.Host", None, http_proxies),
        ("file:///etc/motd", None, http_proxies),
        ("hTTp://u:p@Some.Host/path", "socks5://some.host.proxy", all_proxies),
        ("hTTp://u:p@Other.Host/path", "socks5://http.proxy", all_proxies),
        ("hTTp:///path", "socks5://http.proxy", all_proxies),
        ("hTTps://Other.Host", "socks5://http.proxy", all_proxies),
        ("http://u:p@other.host/path", "http://http.proxy", mixed_proxies),
        ("http://u:p@some.host/path", "http://some.host.proxy", mixed_proxies),
        ("https://u:p@other.host/path", "socks5://http.proxy", mixed_proxies),
        ("https://u:p@some.host/path", "socks5://http.proxy", mixed_proxies),
        ("https://", "socks5://http.proxy", mixed_proxies),
        # XXX: unsure whether this is reasonable behavior
        ("file:///etc/motd", "socks5://http.proxy", all_proxies),
    ),
)
def test_select_proxies(url, expected, proxies):
    """Make sure we can select per-host proxies correctly."""
    assert select_proxy(url, proxies) == expected


@pytest.mark.parametrize(
    "value, expected",
    (
        ('foo="is a fish", bar="as well"', {"foo": "is a fish", "bar": "as well"}),
        ("key_without_value", {"key_without_value": None}),
    ),
)
def test_parse_dict_header(value, expected):
    assert parse_dict_header(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    (
        ("application/xml", ("application/xml", {})),
        (
            "application/json ; charset=utf-8",
            ("application/json", {"charset": "utf-8"}),
        ),
        (
            "application/json ; Charset=utf-8",
            ("application/json", {"charset": "utf-8"}),
        ),
        ("text/plain", ("text/plain", {})),
        (
            "multipart/form-data; boundary = something ; boundary2='something_else' ; no_equals ",
            (
                "multipart/form-data",
                {
                    "boundary": "something",
                    "boundary2": "something_else",
                    "no_equals": True,
                },
            ),
        ),
        (
            'multipart/form-data; boundary = something ; boundary2="something_else" ; no_equals ',
            (
                "multipart/form-data",
                {
                    "boundary": "something",
                    "boundary2": "something_else",
                    "no_equals": True,
                },
            ),
        ),
        (
            "multipart/form-data; boundary = something ; 'boundary2=something_else' ; no_equals ",
            (
                "multipart/form-data",
                {
                    "boundary": "something",
                    "boundary2": "something_else",
                    "no_equals": True,
                },
            ),
        ),
        (
            'multipart/form-data; boundary = something ; "boundary2=something_else" ; no_equals ',
            (
                "multipart/form-data",
                {
                    "boundary": "something",
                    "boundary2": "something_else",
                    "no_equals": True,
                },
            ),
        ),
        ("application/json ; ; ", ("application/json", {})),
    ),
)
def test__parse_content_type_header(value, expected):
    assert _parse_content_type_header(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    (
        (CaseInsensitiveDict(), None),
        (
            CaseInsensitiveDict({"content-type": "application/json; charset=utf-8"}),
            "utf-8",
        ),
        (CaseInsensitiveDict({"content-type": "text/plain"}), "ISO-8859-1"),
    ),
)
def test_get_encoding_from_headers(value, expected):
    assert get_encoding_from_headers(value) == expected


@pytest.mark.parametrize(
    "value, length",
    (
        ("", 0),
        ("T", 1),
        ("Test", 4),
        ("Cont", 0),
        ("Other", -5),
        ("Content", None),
    ),
)
def test_iter_slices(value, length):
    if length is None or (length <= 0 and len(value) > 0):
        # Reads all content at once
        assert len(list(iter_slices(value, length))) == 1
    else:
        assert len(list(iter_slices(value, 1))) == length


@pytest.mark.parametrize(
    "value, expected",
    (
        (
            '<http:/.../front.jpeg>; rel=front; type="image/jpeg"',
            [{"url": "http:/.../front.jpeg", "rel": "front", "type": "image/jpeg"}],
        ),
        ("<http:/.../front.jpeg>", [{"url": "http:/.../front.jpeg"}]),
        ("<http:/.../front.jpeg>;", [{"url": "http:/.../front.jpeg"}]),
        (
            '<http:/.../front.jpeg>; type="image/jpeg",<http://.../back.jpeg>;',
            [
                {"url": "http:/.../front.jpeg", "type": "image/jpeg"},
                {"url": "http://.../back.jpeg"},
            ],
        ),
        ("", []),
    ),
)
def test_parse_header_links(value, expected):
    assert parse_header_links(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    (
        ("example.com/path", "http://example.com/path"),
        ("//example.com/path", "http://example.com/path"),
        ("example.com:80", "http://example.com:80"),
        (
            "http://user:pass@example.com/path?query",
            "http://user:pass@example.com/path?query",
        ),
        ("http://user@example.com/path?query", "http://user@example.com/path?query"),
    ),
)
def test_prepend_scheme_if_needed(value, expected):
    assert prepend_scheme_if_needed(value, "http") == expected


@pytest.mark.parametrize(
    "value, expected",
    (
        ("T", "T"),
        (b"T", "T"),
        ("T", "T"),
    ),
)
def test_to_native_string(value, expected):
    assert to_native_string(value) == expected


@pytest.mark.parametrize(
    "url, expected",
    (
        ("http://u:p@example.com/path?a=1#test", "http://example.com/path?a=1"),
        ("http://example.com/path", "http://example.com/path"),
        ("//u:p@example.com/path", "//example.com/path"),
        ("//example.com/path", "//example.com/path"),
        ("example.com/path", "//example.com/path"),
        ("scheme:u:p@example.com/path", "scheme://example.com/path"),
    ),
)
def test_urldefragauth(url, expected):
    assert urldefragauth(url) == expected


@pytest.mark.parametrize(
    "url, expected",
    (
        ("http://192.168.0.1:5000/", True),
        ("http://192.168.0.1/", True),
        ("http://172.16.1.1/", True),
        ("http://172.16.1.1:5000/", True),
        ("http://localhost.localdomain:5000/v1.0/", True),
        ("http://google.com:6000/", True),
        ("http://172.16.1.12/", False),
        ("http://172.16.1.12:5000/", False),
        ("http://google.com:5000/v1.0/", False),
        ("file:///some/path/on/disk", True),
    ),
)
def test_should_bypass_proxies(url, expected, monkeypatch):
    """Tests for function should_bypass_proxies to check if proxy
    can be bypassed or not
    """
    monkeypatch.setenv(
        "no_proxy",
        "192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1, google.com:6000",
    )
    monkeypatch.setenv(
        "NO_PROXY",
        "192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1, google.com:6000",
    )
    assert should_bypass_proxies(url, no_proxy=None) == expected


@pytest.mark.parametrize(
    "url, expected",
    (
        ("http://172.16.1.1/", "172.16.1.1"),
        ("http://172.16.1.1:5000/", "172.16.1.1"),
        ("http://user:pass@172.16.1.1", "172.16.1.1"),
        ("http://user:pass@172.16.1.1:5000", "172.16.1.1"),
        ("http://hostname/", "hostname"),
        ("http://hostname:5000/", "hostname"),
        ("http://user:pass@hostname", "hostname"),
        ("http://user:pass@hostname:5000", "hostname"),
    ),
)
def test_should_bypass_proxies_pass_only_hostname(url, expected):
    """The proxy_bypass function should be called with a hostname or IP without
    a port number or auth credentials.
    """
    with mock.patch("requests.utils.proxy_bypass") as proxy_bypass:
        should_bypass_proxies(url, no_proxy=None)
        proxy_bypass.assert_called_once_with(expected)


@pytest.mark.parametrize(
    "cookiejar",
    (
        compat.cookielib.CookieJar(),
        RequestsCookieJar(),
    ),
)
def test_add_dict_to_cookiejar(cookiejar):
    """Ensure add_dict_to_cookiejar works for
    non-RequestsCookieJar CookieJars
    """
    cookiedict = {"test": "cookies", "good": "cookies"}
    cj = add_dict_to_cookiejar(cookiejar, cookiedict)
    cookies = {cookie.name: cookie.value for cookie in cj}
    assert cookiedict == cookies


@pytest.mark.parametrize(
    "value, expected",
    (
        ("test", True),
        ("æíöû", False),
        ("ジェーピーニック", False),
    ),
)
def test_unicode_is_ascii(value, expected):
    assert unicode_is_ascii(value) is expected


@pytest.mark.parametrize(
    "url, expected",
    (
        ("http://192.168.0.1:5000/", True),
        ("http://192.168.0.1/", True),
        ("http://172.16.1.1/", True),
        ("http://172.16.1.1:5000/", True),
        ("http://localhost.localdomain:5000/v1.0/", True),
        ("http://172.16.1.12/", False),
        ("http://172.16.1.12:5000/", False),
        ("http://google.com:5000/v1.0/", False),
    ),
)
def test_should_bypass_proxies_no_proxy(url, expected, monkeypatch):
    """Tests for function should_bypass_proxies to check if proxy
    can be bypassed or not using the 'no_proxy' argument
    """
    no_proxy = "192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1"
    # Test 'no_proxy' argument
    assert should_bypass_proxies(url, no_proxy=no_proxy) == expected


@pytest.mark.skipif(os.name != "nt", reason="Test only on Windows")
@pytest.mark.parametrize(
    "url, expected, override",
    (
        ("http://192.168.0.1:5000/", True, None),
        ("http://192.168.0.1/", True, None),
        ("http://172.16.1.1/", True, None),
        ("http://172.16.1.1:5000/", True, None),
        ("http://localhost.localdomain:5000/v1.0/", True, None),
        ("http://172.16.1.22/", False, None),
        ("http://172.16.1.22:5000/", False, None),
        ("http://google.com:5000/v1.0/", False, None),
        ("http://mylocalhostname:5000/v1.0/", True, "<local>"),
        ("http://192.168.0.1/", False, ""),
    ),
)
def test_should_bypass_proxies_win_registry(url, expected, override, monkeypatch):
    """Tests for function should_bypass_proxies to check if proxy
    can be bypassed or not with Windows registry settings
    """
    if override is None:
        override = "192.168.*;127.0.0.1;localhost.localdomain;172.16.1.1"
    import winreg

    class RegHandle:
        def Close(self):
            pass

    ie_settings = RegHandle()
    proxyEnableValues = deque([1, "1"])

    def OpenKey(key, subkey):
        return ie_settings

    def QueryValueEx(key, value_name):
        if key is ie_settings:
            if value_name == "ProxyEnable":
                # this could be a string (REG_SZ) or a 32-bit number (REG_DWORD)
                proxyEnableValues.rotate()
                return [proxyEnableValues[0]]
            elif value_name == "ProxyOverride":
                return [override]

    monkeypatch.setenv("http_proxy", "")
    monkeypatch.setenv("https_proxy", "")
    monkeypatch.setenv("ftp_proxy", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setattr(winreg, "OpenKey", OpenKey)
    monkeypatch.setattr(winreg, "QueryValueEx", QueryValueEx)
    assert should_bypass_proxies(url, None) == expected


@pytest.mark.skipif(os.name != "nt", reason="Test only on Windows")
def test_should_bypass_proxies_win_registry_bad_values(monkeypatch):
    """Tests for function should_bypass_proxies to check if proxy
    can be bypassed or not with Windows invalid registry settings.
    """
    import winreg

    class RegHandle:
        def Close(self):
            pass

    ie_settings = RegHandle()

    def OpenKey(key, subkey):
        return ie_settings

    def QueryValueEx(key, value_name):
        if key is ie_settings:
            if value_name == "ProxyEnable":
                # Invalid response; Should be an int or int-y value
                return [""]
            elif value_name == "ProxyOverride":
                return ["192.168.*;127.0.0.1;localhost.localdomain;172.16.1.1"]

    monkeypatch.setenv("http_proxy", "")
    monkeypatch.setenv("https_proxy", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setattr(winreg, "OpenKey", OpenKey)
    monkeypatch.setattr(winreg, "QueryValueEx", QueryValueEx)
    assert should_bypass_proxies("http://172.16.1.1/", None) is False


@pytest.mark.parametrize(
    "env_name, value",
    (
        ("no_proxy", "192.168.0.0/24,127.0.0.1,localhost.localdomain"),
        ("no_proxy", None),
        ("a_new_key", "192.168.0.0/24,127.0.0.1,localhost.localdomain"),
        ("a_new_key", None),
    ),
)
def test_set_environ(env_name, value):
    """Tests set_environ will set environ values and will restore the environ."""
    environ_copy = copy.deepcopy(os.environ)
    with set_environ(env_name, value):
        assert os.environ.get(env_name) == value

    assert os.environ == environ_copy


def test_set_environ_raises_exception():
    """Tests set_environ will raise exceptions in context when the
    value parameter is None."""
    with pytest.raises(Exception) as exception:
        with set_environ("test1", None):
            raise Exception("Expected exception")

    assert "Expected exception" in str(exception.value)


@pytest.mark.skipif(os.name != "nt", reason="Test only on Windows")
def test_should_bypass_proxies_win_registry_ProxyOverride_value(monkeypatch):
    """Tests for function should_bypass_proxies to check if proxy
    can be bypassed or not with Windows ProxyOverride registry value ending with a semicolon.
    """
    import winreg

    class RegHandle:
        def Close(self):
            pass

    ie_settings = RegHandle()

    def OpenKey(key, subkey):
        return ie_settings

    def QueryValueEx(key, value_name):
        if key is ie_settings:
            if value_name == "ProxyEnable":
                return [1]
            elif value_name == "ProxyOverride":
                return [
                    "192.168.*;127.0.0.1;localhost.localdomain;172.16.1.1;<-loopback>;"
                ]

    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setattr(winreg, "OpenKey", OpenKey)
    monkeypatch.setattr(winreg, "QueryValueEx", QueryValueEx)
    assert should_bypass_proxies("http://example.com/", None) is False


# === tests/conftest.py ===
try:
    from http.server import HTTPServer, SimpleHTTPRequestHandler
except ImportError:
    from BaseHTTPServer import HTTPServer
    from SimpleHTTPServer import SimpleHTTPRequestHandler

import ssl
import threading

import pytest

from requests.compat import urljoin


def prepare_url(value):
    # Issue #1483: Make sure the URL always has a trailing slash
    httpbin_url = value.url.rstrip("/") + "/"

    def inner(*suffix):
        return urljoin(httpbin_url, "/".join(suffix))

    return inner


@pytest.fixture
def httpbin(httpbin):
    return prepare_url(httpbin)


@pytest.fixture
def httpbin_secure(httpbin_secure):
    return prepare_url(httpbin_secure)


@pytest.fixture
def nosan_server(tmp_path_factory):
    # delay importing until the fixture in order to make it possible
    # to deselect the test via command-line when trustme is not available
    import trustme

    tmpdir = tmp_path_factory.mktemp("certs")
    ca = trustme.CA()
    # only commonName, no subjectAltName
    server_cert = ca.issue_cert(common_name="localhost")
    ca_bundle = str(tmpdir / "ca.pem")
    ca.cert_pem.write_to_path(ca_bundle)

    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server_cert.configure_cert(context)
    server = HTTPServer(("localhost", 0), SimpleHTTPRequestHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()

    yield "localhost", server.server_address[1], ca_bundle

    server.shutdown()
    server_thread.join()


# === tests/test_structures.py ===
import pytest

from requests.structures import CaseInsensitiveDict, LookupDict


class TestCaseInsensitiveDict:
    @pytest.fixture(autouse=True)
    def setup(self):
        """CaseInsensitiveDict instance with "Accept" header."""
        self.case_insensitive_dict = CaseInsensitiveDict()
        self.case_insensitive_dict["Accept"] = "application/json"

    def test_list(self):
        assert list(self.case_insensitive_dict) == ["Accept"]

    possible_keys = pytest.mark.parametrize(
        "key", ("accept", "ACCEPT", "aCcEpT", "Accept")
    )

    @possible_keys
    def test_getitem(self, key):
        assert self.case_insensitive_dict[key] == "application/json"

    @possible_keys
    def test_delitem(self, key):
        del self.case_insensitive_dict[key]
        assert key not in self.case_insensitive_dict

    def test_lower_items(self):
        assert list(self.case_insensitive_dict.lower_items()) == [
            ("accept", "application/json")
        ]

    def test_repr(self):
        assert repr(self.case_insensitive_dict) == "{'Accept': 'application/json'}"

    def test_copy(self):
        copy = self.case_insensitive_dict.copy()
        assert copy is not self.case_insensitive_dict
        assert copy == self.case_insensitive_dict

    @pytest.mark.parametrize(
        "other, result",
        (
            ({"AccePT": "application/json"}, True),
            ({}, False),
            (None, False),
        ),
    )
    def test_instance_equality(self, other, result):
        assert (self.case_insensitive_dict == other) is result


class TestLookupDict:
    @pytest.fixture(autouse=True)
    def setup(self):
        """LookupDict instance with "bad_gateway" attribute."""
        self.lookup_dict = LookupDict("test")
        self.lookup_dict.bad_gateway = 502

    def test_repr(self):
        assert repr(self.lookup_dict) == "<lookup 'test'>"

    get_item_parameters = pytest.mark.parametrize(
        "key, value",
        (
            ("bad_gateway", 502),
            ("not_a_key", None),
        ),
    )

    @get_item_parameters
    def test_getitem(self, key, value):
        assert self.lookup_dict[key] == value

    @get_item_parameters
    def test_get(self, key, value):
        assert self.lookup_dict.get(key) == value


# === tests/test_packages.py ===
import requests


def test_can_access_urllib3_attribute():
    requests.packages.urllib3


def test_can_access_idna_attribute():
    requests.packages.idna


def test_can_access_chardet_attribute():
    requests.packages.chardet


# === tests/test_testserver.py ===
import socket
import threading
import time

import pytest
from tests.testserver.server import Server

import requests


class TestTestServer:
    def test_basic(self):
        """messages are sent and received properly"""
        question = b"success?"
        answer = b"yeah, success"

        def handler(sock):
            text = sock.recv(1000)
            assert text == question
            sock.sendall(answer)

        with Server(handler) as (host, port):
            sock = socket.socket()
            sock.connect((host, port))
            sock.sendall(question)
            text = sock.recv(1000)
            assert text == answer
            sock.close()

    def test_server_closes(self):
        """the server closes when leaving the context manager"""
        with Server.basic_response_server() as (host, port):
            sock = socket.socket()
            sock.connect((host, port))

            sock.close()

        with pytest.raises(socket.error):
            new_sock = socket.socket()
            new_sock.connect((host, port))

    def test_text_response(self):
        """the text_response_server sends the given text"""
        server = Server.text_response_server(
            "HTTP/1.1 200 OK\r\n" "Content-Length: 6\r\n" "\r\nroflol"
        )

        with server as (host, port):
            r = requests.get(f"http://{host}:{port}")

            assert r.status_code == 200
            assert r.text == "roflol"
            assert r.headers["Content-Length"] == "6"

    def test_basic_response(self):
        """the basic response server returns an empty http response"""
        with Server.basic_response_server() as (host, port):
            r = requests.get(f"http://{host}:{port}")
            assert r.status_code == 200
            assert r.text == ""
            assert r.headers["Content-Length"] == "0"

    def test_basic_waiting_server(self):
        """the server waits for the block_server event to be set before closing"""
        block_server = threading.Event()

        with Server.basic_response_server(wait_to_close_event=block_server) as (
            host,
            port,
        ):
            sock = socket.socket()
            sock.connect((host, port))
            sock.sendall(b"send something")
            time.sleep(2.5)
            sock.sendall(b"still alive")
            block_server.set()  # release server block

    def test_multiple_requests(self):
        """multiple requests can be served"""
        requests_to_handle = 5

        server = Server.basic_response_server(requests_to_handle=requests_to_handle)

        with server as (host, port):
            server_url = f"http://{host}:{port}"
            for _ in range(requests_to_handle):
                r = requests.get(server_url)
                assert r.status_code == 200

            # the (n+1)th request fails
            with pytest.raises(requests.exceptions.ConnectionError):
                r = requests.get(server_url)

    @pytest.mark.skip(reason="this fails non-deterministically under pytest-xdist")
    def test_request_recovery(self):
        """can check the requests content"""
        # TODO: figure out why this sometimes fails when using pytest-xdist.
        server = Server.basic_response_server(requests_to_handle=2)
        first_request = b"put your hands up in the air"
        second_request = b"put your hand down in the floor"

        with server as address:
            sock1 = socket.socket()
            sock2 = socket.socket()

            sock1.connect(address)
            sock1.sendall(first_request)
            sock1.close()

            sock2.connect(address)
            sock2.sendall(second_request)
            sock2.close()

        assert server.handler_results[0] == first_request
        assert server.handler_results[1] == second_request

    def test_requests_after_timeout_are_not_received(self):
        """the basic response handler times out when receiving requests"""
        server = Server.basic_response_server(request_timeout=1)

        with server as address:
            sock = socket.socket()
            sock.connect(address)
            time.sleep(1.5)
            sock.sendall(b"hehehe, not received")
            sock.close()

        assert server.handler_results[0] == b""

    def test_request_recovery_with_bigger_timeout(self):
        """a biggest timeout can be specified"""
        server = Server.basic_response_server(request_timeout=3)
        data = b"bananadine"

        with server as address:
            sock = socket.socket()
            sock.connect(address)
            time.sleep(1.5)
            sock.sendall(data)
            sock.close()

        assert server.handler_results[0] == data

    def test_server_finishes_on_error(self):
        """the server thread exits even if an exception exits the context manager"""
        server = Server.basic_response_server()
        with pytest.raises(Exception):
            with server:
                raise Exception()

        assert len(server.handler_results) == 0

        # if the server thread fails to finish, the test suite will hang
        # and get killed by the jenkins timeout.

    def test_server_finishes_when_no_connections(self):
        """the server thread exits even if there are no connections"""
        server = Server.basic_response_server()
        with server:
            pass

        assert len(server.handler_results) == 0

        # if the server thread fails to finish, the test suite will hang
        # and get killed by the jenkins timeout.


# === tests/compat.py ===
import warnings

try:
    import StringIO
except ImportError:
    import io as StringIO

try:
    from cStringIO import StringIO as cStringIO
except ImportError:
    cStringIO = None


def u(s):
    warnings.warn(
        (
            "This helper function is no longer relevant in Python 3. "
            "Usage of this alias should be discontinued as it will be "
            "removed in a future release of Requests."
        ),
        DeprecationWarning,
    )
    return s


# === tests/__init__.py ===
"""Requests test package initialisation."""

import warnings

try:
    from urllib3.exceptions import SNIMissingWarning

    # urllib3 1.x sets SNIMissingWarning to only go off once,
    # while this test suite requires it to always fire
    # so that it occurs during test_requests.test_https_warnings
    warnings.simplefilter("always", SNIMissingWarning)
except ImportError:
    # urllib3 2.0 removed that warning and errors out instead
    SNIMissingWarning = None


# === tests/test_hooks.py ===
import pytest

from requests import hooks


def hook(value):
    return value[1:]


@pytest.mark.parametrize(
    "hooks_list, result",
    (
        (hook, "ata"),
        ([hook, lambda x: None, hook], "ta"),
    ),
)
def test_hooks(hooks_list, result):
    assert hooks.dispatch_hook("response", {"response": hooks_list}, "Data") == result


def test_default_hooks():
    assert hooks.default_hooks() == {"response": []}


# === tests/test_requests.py ===
"""Tests for Requests."""

import collections
import contextlib
import io
import json
import os
import pickle
import re
import threading
import warnings
from unittest import mock

import pytest
import urllib3
from urllib3.util import Timeout as Urllib3Timeout

import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth, _basic_auth_str
from requests.compat import (
    JSONDecodeError,
    Morsel,
    MutableMapping,
    builtin_str,
    cookielib,
    getproxies,
    is_urllib3_1,
    urlparse,
)
from requests.cookies import cookiejar_from_dict, morsel_to_cookie
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    ConnectTimeout,
    ContentDecodingError,
    InvalidHeader,
    InvalidProxyURL,
    InvalidSchema,
    InvalidURL,
    MissingSchema,
    ProxyError,
    ReadTimeout,
    RequestException,
    RetryError,
)
from requests.exceptions import SSLError as RequestsSSLError
from requests.exceptions import Timeout, TooManyRedirects, UnrewindableBodyError
from requests.hooks import default_hooks
from requests.models import PreparedRequest, urlencode
from requests.sessions import SessionRedirectMixin
from requests.structures import CaseInsensitiveDict

from . import SNIMissingWarning
from .compat import StringIO
from .testserver.server import TLSServer, consume_socket_content
from .utils import override_environ

# Requests to this URL should always fail with a connection timeout (nothing
# listening on that port)
TARPIT = "http://10.255.255.1"

# This is to avoid waiting the timeout of using TARPIT
INVALID_PROXY = "http://localhost:1"

try:
    from ssl import SSLContext

    del SSLContext
    HAS_MODERN_SSL = True
except ImportError:
    HAS_MODERN_SSL = False

try:
    requests.pyopenssl
    HAS_PYOPENSSL = True
except AttributeError:
    HAS_PYOPENSSL = False


class TestRequests:
    digest_auth_algo = ("MD5", "SHA-256", "SHA-512")

    def test_entry_points(self):
        requests.session
        requests.session().get
        requests.session().head
        requests.get
        requests.head
        requests.put
        requests.patch
        requests.post
        # Not really an entry point, but people rely on it.
        from requests.packages.urllib3.poolmanager import PoolManager  # noqa:F401

    @pytest.mark.parametrize(
        "exception, url",
        (
            (MissingSchema, "hiwpefhipowhefopw"),
            (InvalidSchema, "localhost:3128"),
            (InvalidSchema, "localhost.localdomain:3128/"),
            (InvalidSchema, "10.122.1.1:3128/"),
            (InvalidURL, "http://"),
            (InvalidURL, "http://*example.com"),
            (InvalidURL, "http://.example.com"),
        ),
    )
    def test_invalid_url(self, exception, url):
        with pytest.raises(exception):
            requests.get(url)

    def test_basic_building(self):
        req = requests.Request()
        req.url = "http://kennethreitz.org/"
        req.data = {"life": "42"}

        pr = req.prepare()
        assert pr.url == req.url
        assert pr.body == "life=42"

    @pytest.mark.parametrize("method", ("GET", "HEAD"))
    def test_no_content_length(self, httpbin, method):
        req = requests.Request(method, httpbin(method.lower())).prepare()
        assert "Content-Length" not in req.headers

    @pytest.mark.parametrize("method", ("POST", "PUT", "PATCH", "OPTIONS"))
    def test_no_body_content_length(self, httpbin, method):
        req = requests.Request(method, httpbin(method.lower())).prepare()
        assert req.headers["Content-Length"] == "0"

    @pytest.mark.parametrize("method", ("POST", "PUT", "PATCH", "OPTIONS"))
    def test_empty_content_length(self, httpbin, method):
        req = requests.Request(method, httpbin(method.lower()), data="").prepare()
        assert req.headers["Content-Length"] == "0"

    def test_override_content_length(self, httpbin):
        headers = {"Content-Length": "not zero"}
        r = requests.Request("POST", httpbin("post"), headers=headers).prepare()
        assert "Content-Length" in r.headers
        assert r.headers["Content-Length"] == "not zero"

    def test_path_is_not_double_encoded(self):
        request = requests.Request("GET", "http://0.0.0.0/get/test case").prepare()

        assert request.path_url == "/get/test%20case"

    @pytest.mark.parametrize(
        "url, expected",
        (
            (
                "http://example.com/path#fragment",
                "http://example.com/path?a=b#fragment",
            ),
            (
                "http://example.com/path?key=value#fragment",
                "http://example.com/path?key=value&a=b#fragment",
            ),
        ),
    )
    def test_params_are_added_before_fragment(self, url, expected):
        request = requests.Request("GET", url, params={"a": "b"}).prepare()
        assert request.url == expected

    def test_params_original_order_is_preserved_by_default(self):
        param_ordered_dict = collections.OrderedDict(
            (("z", 1), ("a", 1), ("k", 1), ("d", 1))
        )
        session = requests.Session()
        request = requests.Request(
            "GET", "http://example.com/", params=param_ordered_dict
        )
        prep = session.prepare_request(request)
        assert prep.url == "http://example.com/?z=1&a=1&k=1&d=1"

    def test_params_bytes_are_encoded(self):
        request = requests.Request(
            "GET", "http://example.com", params=b"test=foo"
        ).prepare()
        assert request.url == "http://example.com/?test=foo"

    def test_binary_put(self):
        request = requests.Request(
            "PUT", "http://example.com", data="ööö".encode()
        ).prepare()
        assert isinstance(request.body, bytes)

    def test_whitespaces_are_removed_from_url(self):
        # Test for issue #3696
        request = requests.Request("GET", " http://example.com").prepare()
        assert request.url == "http://example.com/"

    @pytest.mark.parametrize("scheme", ("http://", "HTTP://", "hTTp://", "HttP://"))
    def test_mixed_case_scheme_acceptable(self, httpbin, scheme):
        s = requests.Session()
        s.proxies = getproxies()
        parts = urlparse(httpbin("get"))
        url = scheme + parts.netloc + parts.path
        r = requests.Request("GET", url)
        r = s.send(r.prepare())
        assert r.status_code == 200, f"failed for scheme {scheme}"

    def test_HTTP_200_OK_GET_ALTERNATIVE(self, httpbin):
        r = requests.Request("GET", httpbin("get"))
        s = requests.Session()
        s.proxies = getproxies()

        r = s.send(r.prepare())

        assert r.status_code == 200

    def test_HTTP_302_ALLOW_REDIRECT_GET(self, httpbin):
        r = requests.get(httpbin("redirect", "1"))
        assert r.status_code == 200
        assert r.history[0].status_code == 302
        assert r.history[0].is_redirect

    def test_HTTP_307_ALLOW_REDIRECT_POST(self, httpbin):
        r = requests.post(
            httpbin("redirect-to"),
            data="test",
            params={"url": "post", "status_code": 307},
        )
        assert r.status_code == 200
        assert r.history[0].status_code == 307
        assert r.history[0].is_redirect
        assert r.json()["data"] == "test"

    def test_HTTP_307_ALLOW_REDIRECT_POST_WITH_SEEKABLE(self, httpbin):
        byte_str = b"test"
        r = requests.post(
            httpbin("redirect-to"),
            data=io.BytesIO(byte_str),
            params={"url": "post", "status_code": 307},
        )
        assert r.status_code == 200
        assert r.history[0].status_code == 307
        assert r.history[0].is_redirect
        assert r.json()["data"] == byte_str.decode("utf-8")

    def test_HTTP_302_TOO_MANY_REDIRECTS(self, httpbin):
        try:
            requests.get(httpbin("relative-redirect", "50"))
        except TooManyRedirects as e:
            url = httpbin("relative-redirect", "20")
            assert e.request.url == url
            assert e.response.url == url
            assert len(e.response.history) == 30
        else:
            pytest.fail("Expected redirect to raise TooManyRedirects but it did not")

    def test_HTTP_302_TOO_MANY_REDIRECTS_WITH_PARAMS(self, httpbin):
        s = requests.session()
        s.max_redirects = 5
        try:
            s.get(httpbin("relative-redirect", "50"))
        except TooManyRedirects as e:
            url = httpbin("relative-redirect", "45")
            assert e.request.url == url
            assert e.response.url == url
            assert len(e.response.history) == 5
        else:
            pytest.fail(
                "Expected custom max number of redirects to be respected but was not"
            )

    def test_http_301_changes_post_to_get(self, httpbin):
        r = requests.post(httpbin("status", "301"))
        assert r.status_code == 200
        assert r.request.method == "GET"
        assert r.history[0].status_code == 301
        assert r.history[0].is_redirect

    def test_http_301_doesnt_change_head_to_get(self, httpbin):
        r = requests.head(httpbin("status", "301"), allow_redirects=True)
        print(r.content)
        assert r.status_code == 200
        assert r.request.method == "HEAD"
        assert r.history[0].status_code == 301
        assert r.history[0].is_redirect

    def test_http_302_changes_post_to_get(self, httpbin):
        r = requests.post(httpbin("status", "302"))
        assert r.status_code == 200
        assert r.request.method == "GET"
        assert r.history[0].status_code == 302
        assert r.history[0].is_redirect

    def test_http_302_doesnt_change_head_to_get(self, httpbin):
        r = requests.head(httpbin("status", "302"), allow_redirects=True)
        assert r.status_code == 200
        assert r.request.method == "HEAD"
        assert r.history[0].status_code == 302
        assert r.history[0].is_redirect

    def test_http_303_changes_post_to_get(self, httpbin):
        r = requests.post(httpbin("status", "303"))
        assert r.status_code == 200
        assert r.request.method == "GET"
        assert r.history[0].status_code == 303
        assert r.history[0].is_redirect

    def test_http_303_doesnt_change_head_to_get(self, httpbin):
        r = requests.head(httpbin("status", "303"), allow_redirects=True)
        assert r.status_code == 200
        assert r.request.method == "HEAD"
        assert r.history[0].status_code == 303
        assert r.history[0].is_redirect

    def test_header_and_body_removal_on_redirect(self, httpbin):
        purged_headers = ("Content-Length", "Content-Type")
        ses = requests.Session()
        req = requests.Request("POST", httpbin("post"), data={"test": "data"})
        prep = ses.prepare_request(req)
        resp = ses.send(prep)

        # Mimic a redirect response
        resp.status_code = 302
        resp.headers["location"] = "get"

        # Run request through resolve_redirects
        next_resp = next(ses.resolve_redirects(resp, prep))
        assert next_resp.request.body is None
        for header in purged_headers:
            assert header not in next_resp.request.headers

    def test_transfer_enc_removal_on_redirect(self, httpbin):
        purged_headers = ("Transfer-Encoding", "Content-Type")
        ses = requests.Session()
        req = requests.Request("POST", httpbin("post"), data=(b"x" for x in range(1)))
        prep = ses.prepare_request(req)
        assert "Transfer-Encoding" in prep.headers

        # Create Response to avoid https://github.com/kevin1024/pytest-httpbin/issues/33
        resp = requests.Response()
        resp.raw = io.BytesIO(b"the content")
        resp.request = prep
        setattr(resp.raw, "release_conn", lambda *args: args)

        # Mimic a redirect response
        resp.status_code = 302
        resp.headers["location"] = httpbin("get")

        # Run request through resolve_redirect
        next_resp = next(ses.resolve_redirects(resp, prep))
        assert next_resp.request.body is None
        for header in purged_headers:
            assert header not in next_resp.request.headers

    def test_fragment_maintained_on_redirect(self, httpbin):
        fragment = "#view=edit&token=hunter2"
        r = requests.get(httpbin("redirect-to?url=get") + fragment)

        assert len(r.history) > 0
        assert r.history[0].request.url == httpbin("redirect-to?url=get") + fragment
        assert r.url == httpbin("get") + fragment

    def test_HTTP_200_OK_GET_WITH_PARAMS(self, httpbin):
        heads = {"User-agent": "Mozilla/5.0"}

        r = requests.get(httpbin("user-agent"), headers=heads)

        assert heads["User-agent"] in r.text
        assert r.status_code == 200

    def test_HTTP_200_OK_GET_WITH_MIXED_PARAMS(self, httpbin):
        heads = {"User-agent": "Mozilla/5.0"}

        r = requests.get(
            httpbin("get") + "?test=true", params={"q": "test"}, headers=heads
        )
        assert r.status_code == 200

    def test_set_cookie_on_301(self, httpbin):
        s = requests.session()
        url = httpbin("cookies/set?foo=bar")
        s.get(url)
        assert s.cookies["foo"] == "bar"

    def test_cookie_sent_on_redirect(self, httpbin):
        s = requests.session()
        s.get(httpbin("cookies/set?foo=bar"))
        r = s.get(httpbin("redirect/1"))  # redirects to httpbin('get')
        assert "Cookie" in r.json()["headers"]

    def test_cookie_removed_on_expire(self, httpbin):
        s = requests.session()
        s.get(httpbin("cookies/set?foo=bar"))
        assert s.cookies["foo"] == "bar"
        s.get(
            httpbin("response-headers"),
            params={"Set-Cookie": "foo=deleted; expires=Thu, 01-Jan-1970 00:00:01 GMT"},
        )
        assert "foo" not in s.cookies

    def test_cookie_quote_wrapped(self, httpbin):
        s = requests.session()
        s.get(httpbin('cookies/set?foo="bar:baz"'))
        assert s.cookies["foo"] == '"bar:baz"'

    def test_cookie_persists_via_api(self, httpbin):
        s = requests.session()
        r = s.get(httpbin("redirect/1"), cookies={"foo": "bar"})
        assert "foo" in r.request.headers["Cookie"]
        assert "foo" in r.history[0].request.headers["Cookie"]

    def test_request_cookie_overrides_session_cookie(self, httpbin):
        s = requests.session()
        s.cookies["foo"] = "bar"
        r = s.get(httpbin("cookies"), cookies={"foo": "baz"})
        assert r.json()["cookies"]["foo"] == "baz"
        # Session cookie should not be modified
        assert s.cookies["foo"] == "bar"

    def test_request_cookies_not_persisted(self, httpbin):
        s = requests.session()
        s.get(httpbin("cookies"), cookies={"foo": "baz"})
        # Sending a request with cookies should not add cookies to the session
        assert not s.cookies

    def test_generic_cookiejar_works(self, httpbin):
        cj = cookielib.CookieJar()
        cookiejar_from_dict({"foo": "bar"}, cj)
        s = requests.session()
        s.cookies = cj
        r = s.get(httpbin("cookies"))
        # Make sure the cookie was sent
        assert r.json()["cookies"]["foo"] == "bar"
        # Make sure the session cj is still the custom one
        assert s.cookies is cj

    def test_param_cookiejar_works(self, httpbin):
        cj = cookielib.CookieJar()
        cookiejar_from_dict({"foo": "bar"}, cj)
        s = requests.session()
        r = s.get(httpbin("cookies"), cookies=cj)
        # Make sure the cookie was sent
        assert r.json()["cookies"]["foo"] == "bar"

    def test_cookielib_cookiejar_on_redirect(self, httpbin):
        """Tests resolve_redirect doesn't fail when merging cookies
        with non-RequestsCookieJar cookiejar.

        See GH #3579
        """
        cj = cookiejar_from_dict({"foo": "bar"}, cookielib.CookieJar())
        s = requests.Session()
        s.cookies = cookiejar_from_dict({"cookie": "tasty"})

        # Prepare request without using Session
        req = requests.Request("GET", httpbin("headers"), cookies=cj)
        prep_req = req.prepare()

        # Send request and simulate redirect
        resp = s.send(prep_req)
        resp.status_code = 302
        resp.headers["location"] = httpbin("get")
        redirects = s.resolve_redirects(resp, prep_req)
        resp = next(redirects)

        # Verify CookieJar isn't being converted to RequestsCookieJar
        assert isinstance(prep_req._cookies, cookielib.CookieJar)
        assert isinstance(resp.request._cookies, cookielib.CookieJar)
        assert not isinstance(resp.request._cookies, requests.cookies.RequestsCookieJar)

        cookies = {}
        for c in resp.request._cookies:
            cookies[c.name] = c.value
        assert cookies["foo"] == "bar"
        assert cookies["cookie"] == "tasty"

    def test_requests_in_history_are_not_overridden(self, httpbin):
        resp = requests.get(httpbin("redirect/3"))
        urls = [r.url for r in resp.history]
        req_urls = [r.request.url for r in resp.history]
        assert urls == req_urls

    def test_history_is_always_a_list(self, httpbin):
        """Show that even with redirects, Response.history is always a list."""
        resp = requests.get(httpbin("get"))
        assert isinstance(resp.history, list)
        resp = requests.get(httpbin("redirect/1"))
        assert isinstance(resp.history, list)
        assert not isinstance(resp.history, tuple)

    def test_headers_on_session_with_None_are_not_sent(self, httpbin):
        """Do not send headers in Session.headers with None values."""
        ses = requests.Session()
        ses.headers["Accept-Encoding"] = None
        req = requests.Request("GET", httpbin("get"))
        prep = ses.prepare_request(req)
        assert "Accept-Encoding" not in prep.headers

    def test_headers_preserve_order(self, httpbin):
        """Preserve order when headers provided as OrderedDict."""
        ses = requests.Session()
        ses.headers = collections.OrderedDict()
        ses.headers["Accept-Encoding"] = "identity"
        ses.headers["First"] = "1"
        ses.headers["Second"] = "2"
        headers = collections.OrderedDict([("Third", "3"), ("Fourth", "4")])
        headers["Fifth"] = "5"
        headers["Second"] = "222"
        req = requests.Request("GET", httpbin("get"), headers=headers)
        prep = ses.prepare_request(req)
        items = list(prep.headers.items())
        assert items[0] == ("Accept-Encoding", "identity")
        assert items[1] == ("First", "1")
        assert items[2] == ("Second", "222")
        assert items[3] == ("Third", "3")
        assert items[4] == ("Fourth", "4")
        assert items[5] == ("Fifth", "5")

    @pytest.mark.parametrize("key", ("User-agent", "user-agent"))
    def test_user_agent_transfers(self, httpbin, key):
        heads = {key: "Mozilla/5.0 (github.com/psf/requests)"}

        r = requests.get(httpbin("user-agent"), headers=heads)
        assert heads[key] in r.text

    def test_HTTP_200_OK_HEAD(self, httpbin):
        r = requests.head(httpbin("get"))
        assert r.status_code == 200

    def test_HTTP_200_OK_PUT(self, httpbin):
        r = requests.put(httpbin("put"))
        assert r.status_code == 200

    def test_BASICAUTH_TUPLE_HTTP_200_OK_GET(self, httpbin):
        auth = ("user", "pass")
        url = httpbin("basic-auth", "user", "pass")

        r = requests.get(url, auth=auth)
        assert r.status_code == 200

        r = requests.get(url)
        assert r.status_code == 401

        s = requests.session()
        s.auth = auth
        r = s.get(url)
        assert r.status_code == 200

    @pytest.mark.parametrize(
        "username, password",
        (
            ("user", "pass"),
            ("имя".encode(), "пароль".encode()),
            (42, 42),
            (None, None),
        ),
    )
    def test_set_basicauth(self, httpbin, username, password):
        auth = (username, password)
        url = httpbin("get")

        r = requests.Request("GET", url, auth=auth)
        p = r.prepare()

        assert p.headers["Authorization"] == _basic_auth_str(username, password)

    def test_basicauth_encodes_byte_strings(self):
        """Ensure b'test' formats as the byte string "test" rather
        than the unicode string "b'test'" in Python 3.
        """
        auth = (b"\xc5\xafsername", b"test\xc6\xb6")
        r = requests.Request("GET", "http://localhost", auth=auth)
        p = r.prepare()

        assert p.headers["Authorization"] == "Basic xa9zZXJuYW1lOnRlc3TGtg=="

    @pytest.mark.parametrize(
        "url, exception",
        (
            # Connecting to an unknown domain should raise a ConnectionError
            ("http://doesnotexist.google.com", ConnectionError),
            # Connecting to an invalid port should raise a ConnectionError
            ("http://localhost:1", ConnectionError),
            # Inputing a URL that cannot be parsed should raise an InvalidURL error
            ("http://fe80::5054:ff:fe5a:fc0", InvalidURL),
        ),
    )
    def test_errors(self, url, exception):
        with pytest.raises(exception):
            requests.get(url, timeout=1)

    def test_proxy_error(self):
        # any proxy related error (address resolution, no route to host, etc) should result in a ProxyError
        with pytest.raises(ProxyError):
            requests.get(
                "http://localhost:1", proxies={"http": "non-resolvable-address"}
            )

    def test_proxy_error_on_bad_url(self, httpbin, httpbin_secure):
        with pytest.raises(InvalidProxyURL):
            requests.get(httpbin_secure(), proxies={"https": "http:/badproxyurl:3128"})

        with pytest.raises(InvalidProxyURL):
            requests.get(httpbin(), proxies={"http": "http://:8080"})

        with pytest.raises(InvalidProxyURL):
            requests.get(httpbin_secure(), proxies={"https": "https://"})

        with pytest.raises(InvalidProxyURL):
            requests.get(httpbin(), proxies={"http": "http:///example.com:8080"})

    def test_respect_proxy_env_on_send_self_prepared_request(self, httpbin):
        with override_environ(http_proxy=INVALID_PROXY):
            with pytest.raises(ProxyError):
                session = requests.Session()
                request = requests.Request("GET", httpbin())
                session.send(request.prepare())

    def test_respect_proxy_env_on_send_session_prepared_request(self, httpbin):
        with override_environ(http_proxy=INVALID_PROXY):
            with pytest.raises(ProxyError):
                session = requests.Session()
                request = requests.Request("GET", httpbin())
                prepared = session.prepare_request(request)
                session.send(prepared)

    def test_respect_proxy_env_on_send_with_redirects(self, httpbin):
        with override_environ(http_proxy=INVALID_PROXY):
            with pytest.raises(ProxyError):
                session = requests.Session()
                url = httpbin("redirect/1")
                print(url)
                request = requests.Request("GET", url)
                session.send(request.prepare())

    def test_respect_proxy_env_on_get(self, httpbin):
        with override_environ(http_proxy=INVALID_PROXY):
            with pytest.raises(ProxyError):
                session = requests.Session()
                session.get(httpbin())

    def test_respect_proxy_env_on_request(self, httpbin):
        with override_environ(http_proxy=INVALID_PROXY):
            with pytest.raises(ProxyError):
                session = requests.Session()
                session.request(method="GET", url=httpbin())

    def test_proxy_authorization_preserved_on_request(self, httpbin):
        proxy_auth_value = "Bearer XXX"
        session = requests.Session()
        session.headers.update({"Proxy-Authorization": proxy_auth_value})
        resp = session.request(method="GET", url=httpbin("get"))
        sent_headers = resp.json().get("headers", {})

        assert sent_headers.get("Proxy-Authorization") == proxy_auth_value

    @pytest.mark.parametrize(
        "url,has_proxy_auth",
        (
            ("http://example.com", True),
            ("https://example.com", False),
        ),
    )
    def test_proxy_authorization_not_appended_to_https_request(
        self, url, has_proxy_auth
    ):
        session = requests.Session()
        proxies = {
            "http": "http://test:pass@localhost:8080",
            "https": "http://test:pass@localhost:8090",
        }
        req = requests.Request("GET", url)
        prep = req.prepare()
        session.rebuild_proxies(prep, proxies)

        assert ("Proxy-Authorization" in prep.headers) is has_proxy_auth

    def test_basicauth_with_netrc(self, httpbin):
        auth = ("user", "pass")
        wrong_auth = ("wronguser", "wrongpass")
        url = httpbin("basic-auth", "user", "pass")

        old_auth = requests.sessions.get_netrc_auth

        try:

            def get_netrc_auth_mock(url):
                return auth

            requests.sessions.get_netrc_auth = get_netrc_auth_mock

            # Should use netrc and work.
            r = requests.get(url)
            assert r.status_code == 200

            # Given auth should override and fail.
            r = requests.get(url, auth=wrong_auth)
            assert r.status_code == 401

            s = requests.session()

            # Should use netrc and work.
            r = s.get(url)
            assert r.status_code == 200

            # Given auth should override and fail.
            s.auth = wrong_auth
            r = s.get(url)
            assert r.status_code == 401
        finally:
            requests.sessions.get_netrc_auth = old_auth

    def test_DIGEST_HTTP_200_OK_GET(self, httpbin):
        for authtype in self.digest_auth_algo:
            auth = HTTPDigestAuth("user", "pass")
            url = httpbin("digest-auth", "auth", "user", "pass", authtype, "never")

            r = requests.get(url, auth=auth)
            assert r.status_code == 200

            r = requests.get(url)
            assert r.status_code == 401
            print(r.headers["WWW-Authenticate"])

            s = requests.session()
            s.auth = HTTPDigestAuth("user", "pass")
            r = s.get(url)
            assert r.status_code == 200

    def test_DIGEST_AUTH_RETURNS_COOKIE(self, httpbin):
        for authtype in self.digest_auth_algo:
            url = httpbin("digest-auth", "auth", "user", "pass", authtype)
            auth = HTTPDigestAuth("user", "pass")
            r = requests.get(url)
            assert r.cookies["fake"] == "fake_value"

            r = requests.get(url, auth=auth)
            assert r.status_code == 200

    def test_DIGEST_AUTH_SETS_SESSION_COOKIES(self, httpbin):
        for authtype in self.digest_auth_algo:
            url = httpbin("digest-auth", "auth", "user", "pass", authtype)
            auth = HTTPDigestAuth("user", "pass")
            s = requests.Session()
            s.get(url, auth=auth)
            assert s.cookies["fake"] == "fake_value"

    def test_DIGEST_STREAM(self, httpbin):
        for authtype in self.digest_auth_algo:
            auth = HTTPDigestAuth("user", "pass")
            url = httpbin("digest-auth", "auth", "user", "pass", authtype)

            r = requests.get(url, auth=auth, stream=True)
            assert r.raw.read() != b""

            r = requests.get(url, auth=auth, stream=False)
            assert r.raw.read() == b""

    def test_DIGESTAUTH_WRONG_HTTP_401_GET(self, httpbin):
        for authtype in self.digest_auth_algo:
            auth = HTTPDigestAuth("user", "wrongpass")
            url = httpbin("digest-auth", "auth", "user", "pass", authtype)

            r = requests.get(url, auth=auth)
            assert r.status_code == 401

            r = requests.get(url)
            assert r.status_code == 401

            s = requests.session()
            s.auth = auth
            r = s.get(url)
            assert r.status_code == 401

    def test_DIGESTAUTH_QUOTES_QOP_VALUE(self, httpbin):
        for authtype in self.digest_auth_algo:
            auth = HTTPDigestAuth("user", "pass")
            url = httpbin("digest-auth", "auth", "user", "pass", authtype)

            r = requests.get(url, auth=auth)
            assert '"auth"' in r.request.headers["Authorization"]

    def test_POSTBIN_GET_POST_FILES(self, httpbin):
        url = httpbin("post")
        requests.post(url).raise_for_status()

        post1 = requests.post(url, data={"some": "data"})
        assert post1.status_code == 200

        with open("requirements-dev.txt") as f:
            post2 = requests.post(url, files={"some": f})
        assert post2.status_code == 200

        post4 = requests.post(url, data='[{"some": "json"}]')
        assert post4.status_code == 200

        with pytest.raises(ValueError):
            requests.post(url, files=["bad file data"])

    def test_invalid_files_input(self, httpbin):
        url = httpbin("post")
        post = requests.post(url, files={"random-file-1": None, "random-file-2": 1})
        assert b'name="random-file-1"' not in post.request.body
        assert b'name="random-file-2"' in post.request.body

    def test_POSTBIN_SEEKED_OBJECT_WITH_NO_ITER(self, httpbin):
        class TestStream:
            def __init__(self, data):
                self.data = data.encode()
                self.length = len(self.data)
                self.index = 0

            def __len__(self):
                return self.length

            def read(self, size=None):
                if size:
                    ret = self.data[self.index : self.index + size]
                    self.index += size
                else:
                    ret = self.data[self.index :]
                    self.index = self.length
                return ret

            def tell(self):
                return self.index

            def seek(self, offset, where=0):
                if where == 0:
                    self.index = offset
                elif where == 1:
                    self.index += offset
                elif where == 2:
                    self.index = self.length + offset

        test = TestStream("test")
        post1 = requests.post(httpbin("post"), data=test)
        assert post1.status_code == 200
        assert post1.json()["data"] == "test"

        test = TestStream("test")
        test.seek(2)
        post2 = requests.post(httpbin("post"), data=test)
        assert post2.status_code == 200
        assert post2.json()["data"] == "st"

    def test_POSTBIN_GET_POST_FILES_WITH_DATA(self, httpbin):
        url = httpbin("post")
        requests.post(url).raise_for_status()

        post1 = requests.post(url, data={"some": "data"})
        assert post1.status_code == 200

        with open("requirements-dev.txt") as f:
            post2 = requests.post(url, data={"some": "data"}, files={"some": f})
        assert post2.status_code == 200

        post4 = requests.post(url, data='[{"some": "json"}]')
        assert post4.status_code == 200

        with pytest.raises(ValueError):
            requests.post(url, files=["bad file data"])

    def test_post_with_custom_mapping(self, httpbin):
        class CustomMapping(MutableMapping):
            def __init__(self, *args, **kwargs):
                self.data = dict(*args, **kwargs)

            def __delitem__(self, key):
                del self.data[key]

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.data[key] = value

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        data = CustomMapping({"some": "data"})
        url = httpbin("post")
        found_json = requests.post(url, data=data).json().get("form")
        assert found_json == {"some": "data"}

    def test_conflicting_post_params(self, httpbin):
        url = httpbin("post")
        with open("requirements-dev.txt") as f:
            with pytest.raises(ValueError):
                requests.post(url, data='[{"some": "data"}]', files={"some": f})

    def test_request_ok_set(self, httpbin):
        r = requests.get(httpbin("status", "404"))
        assert not r.ok

    def test_status_raising(self, httpbin):
        r = requests.get(httpbin("status", "404"))
        with pytest.raises(requests.exceptions.HTTPError):
            r.raise_for_status()

        r = requests.get(httpbin("status", "500"))
        assert not r.ok

    def test_decompress_gzip(self, httpbin):
        r = requests.get(httpbin("gzip"))
        r.content.decode("ascii")

    @pytest.mark.parametrize(
        "url, params",
        (
            ("/get", {"foo": "føø"}),
            ("/get", {"føø": "føø"}),
            ("/get", {"føø": "føø"}),
            ("/get", {"foo": "foo"}),
            ("ø", {"foo": "foo"}),
        ),
    )
    def test_unicode_get(self, httpbin, url, params):
        requests.get(httpbin(url), params=params)

    def test_unicode_header_name(self, httpbin):
        requests.put(
            httpbin("put"),
            headers={"Content-Type": "application/octet-stream"},
            data="\xff",
        )  # compat.str is unicode.

    def test_pyopenssl_redirect(self, httpbin_secure, httpbin_ca_bundle):
        requests.get(httpbin_secure("status", "301"), verify=httpbin_ca_bundle)

    def test_invalid_ca_certificate_path(self, httpbin_secure):
        INVALID_PATH = "/garbage"
        with pytest.raises(IOError) as e:
            requests.get(httpbin_secure(), verify=INVALID_PATH)
        assert str(
            e.value
        ) == "Could not find a suitable TLS CA certificate bundle, invalid path: {}".format(
            INVALID_PATH
        )

    def test_invalid_ssl_certificate_files(self, httpbin_secure):
        INVALID_PATH = "/garbage"
        with pytest.raises(IOError) as e:
            requests.get(httpbin_secure(), cert=INVALID_PATH)
        assert str(
            e.value
        ) == "Could not find the TLS certificate file, invalid path: {}".format(
            INVALID_PATH
        )

        with pytest.raises(IOError) as e:
            requests.get(httpbin_secure(), cert=(".", INVALID_PATH))
        assert str(e.value) == (
            f"Could not find the TLS key file, invalid path: {INVALID_PATH}"
        )

    @pytest.mark.parametrize(
        "env, expected",
        (
            ({}, True),
            ({"REQUESTS_CA_BUNDLE": "/some/path"}, "/some/path"),
            ({"REQUESTS_CA_BUNDLE": ""}, True),
            ({"CURL_CA_BUNDLE": "/some/path"}, "/some/path"),
            ({"CURL_CA_BUNDLE": ""}, True),
            ({"REQUESTS_CA_BUNDLE": "", "CURL_CA_BUNDLE": ""}, True),
            (
                {
                    "REQUESTS_CA_BUNDLE": "/some/path",
                    "CURL_CA_BUNDLE": "/curl/path",
                },
                "/some/path",
            ),
            (
                {
                    "REQUESTS_CA_BUNDLE": "",
                    "CURL_CA_BUNDLE": "/curl/path",
                },
                "/curl/path",
            ),
        ),
    )
    def test_env_cert_bundles(self, httpbin, env, expected):
        s = requests.Session()
        with mock.patch("os.environ", env):
            settings = s.merge_environment_settings(
                url=httpbin("get"), proxies={}, stream=False, verify=True, cert=None
            )
        assert settings["verify"] == expected

    def test_http_with_certificate(self, httpbin):
        r = requests.get(httpbin(), cert=".")
        assert r.status_code == 200

    @pytest.mark.skipif(
        SNIMissingWarning is None,
        reason="urllib3 2.0 removed that warning and errors out instead",
    )
    def test_https_warnings(self, nosan_server):
        """warnings are emitted with requests.get"""
        host, port, ca_bundle = nosan_server
        if HAS_MODERN_SSL or HAS_PYOPENSSL:
            warnings_expected = ("SubjectAltNameWarning",)
        else:
            warnings_expected = (
                "SNIMissingWarning",
                "InsecurePlatformWarning",
                "SubjectAltNameWarning",
            )

        with pytest.warns() as warning_records:
            warnings.simplefilter("always")
            requests.get(f"https://localhost:{port}/", verify=ca_bundle)

        warning_records = [
            item
            for item in warning_records
            if item.category.__name__ != "ResourceWarning"
        ]

        warnings_category = tuple(item.category.__name__ for item in warning_records)
        assert warnings_category == warnings_expected

    def test_certificate_failure(self, httpbin_secure):
        """
        When underlying SSL problems occur, an SSLError is raised.
        """
        with pytest.raises(RequestsSSLError):
            # Our local httpbin does not have a trusted CA, so this call will
            # fail if we use our default trust bundle.
            requests.get(httpbin_secure("status", "200"))

    def test_urlencoded_get_query_multivalued_param(self, httpbin):
        r = requests.get(httpbin("get"), params={"test": ["foo", "baz"]})
        assert r.status_code == 200
        assert r.url == httpbin("get?test=foo&test=baz")

    def test_form_encoded_post_query_multivalued_element(self, httpbin):
        r = requests.Request(
            method="POST", url=httpbin("post"), data=dict(test=["foo", "baz"])
        )
        prep = r.prepare()
        assert prep.body == "test=foo&test=baz"

    def test_different_encodings_dont_break_post(self, httpbin):
        with open(__file__, "rb") as f:
            r = requests.post(
                httpbin("post"),
                data={"stuff": json.dumps({"a": 123})},
                params={"blah": "asdf1234"},
                files={"file": ("test_requests.py", f)},
            )
        assert r.status_code == 200

    @pytest.mark.parametrize(
        "data",
        (
            {"stuff": "ëlïxr"},
            {"stuff": "ëlïxr".encode()},
            {"stuff": "elixr"},
            {"stuff": b"elixr"},
        ),
    )
    def test_unicode_multipart_post(self, httpbin, data):
        with open(__file__, "rb") as f:
            r = requests.post(
                httpbin("post"),
                data=data,
                files={"file": ("test_requests.py", f)},
            )
        assert r.status_code == 200

    def test_unicode_multipart_post_fieldnames(self, httpbin):
        filename = os.path.splitext(__file__)[0] + ".py"
        with open(filename, "rb") as f:
            r = requests.Request(
                method="POST",
                url=httpbin("post"),
                data={b"stuff": "elixr"},
                files={"file": ("test_requests.py", f)},
            )
            prep = r.prepare()

        assert b'name="stuff"' in prep.body
        assert b"name=\"b'stuff'\"" not in prep.body

    def test_unicode_method_name(self, httpbin):
        with open(__file__, "rb") as f:
            files = {"file": f}
            r = requests.request(
                method="POST",
                url=httpbin("post"),
                files=files,
            )
        assert r.status_code == 200

    def test_unicode_method_name_with_request_object(self, httpbin):
        s = requests.Session()
        with open(__file__, "rb") as f:
            files = {"file": f}
            req = requests.Request("POST", httpbin("post"), files=files)
            prep = s.prepare_request(req)
        assert isinstance(prep.method, builtin_str)
        assert prep.method == "POST"

        resp = s.send(prep)
        assert resp.status_code == 200

    def test_non_prepared_request_error(self):
        s = requests.Session()
        req = requests.Request("POST", "/")

        with pytest.raises(ValueError) as e:
            s.send(req)
        assert str(e.value) == "You can only send PreparedRequests."

    def test_custom_content_type(self, httpbin):
        with open(__file__, "rb") as f1:
            with open(__file__, "rb") as f2:
                data = {"stuff": json.dumps({"a": 123})}
                files = {
                    "file1": ("test_requests.py", f1),
                    "file2": ("test_requests", f2, "text/py-content-type"),
                }
                r = requests.post(httpbin("post"), data=data, files=files)
        assert r.status_code == 200
        assert b"text/py-content-type" in r.request.body

    def test_hook_receives_request_arguments(self, httpbin):
        def hook(resp, **kwargs):
            assert resp is not None
            assert kwargs != {}

        s = requests.Session()
        r = requests.Request("GET", httpbin(), hooks={"response": hook})
        prep = s.prepare_request(r)
        s.send(prep)

    def test_session_hooks_are_used_with_no_request_hooks(self, httpbin):
        def hook(*args, **kwargs):
            pass

        s = requests.Session()
        s.hooks["response"].append(hook)
        r = requests.Request("GET", httpbin())
        prep = s.prepare_request(r)
        assert prep.hooks["response"] != []
        assert prep.hooks["response"] == [hook]

    def test_session_hooks_are_overridden_by_request_hooks(self, httpbin):
        def hook1(*args, **kwargs):
            pass

        def hook2(*args, **kwargs):
            pass

        assert hook1 is not hook2
        s = requests.Session()
        s.hooks["response"].append(hook2)
        r = requests.Request("GET", httpbin(), hooks={"response": [hook1]})
        prep = s.prepare_request(r)
        assert prep.hooks["response"] == [hook1]

    def test_prepared_request_hook(self, httpbin):
        def hook(resp, **kwargs):
            resp.hook_working = True
            return resp

        req = requests.Request("GET", httpbin(), hooks={"response": hook})
        prep = req.prepare()

        s = requests.Session()
        s.proxies = getproxies()
        resp = s.send(prep)

        assert hasattr(resp, "hook_working")

    def test_prepared_from_session(self, httpbin):
        class DummyAuth(requests.auth.AuthBase):
            def __call__(self, r):
                r.headers["Dummy-Auth-Test"] = "dummy-auth-test-ok"
                return r

        req = requests.Request("GET", httpbin("headers"))
        assert not req.auth

        s = requests.Session()
        s.auth = DummyAuth()

        prep = s.prepare_request(req)
        resp = s.send(prep)

        assert resp.json()["headers"]["Dummy-Auth-Test"] == "dummy-auth-test-ok"

    def test_prepare_request_with_bytestring_url(self):
        req = requests.Request("GET", b"https://httpbin.org/")
        s = requests.Session()
        prep = s.prepare_request(req)
        assert prep.url == "https://httpbin.org/"

    def test_request_with_bytestring_host(self, httpbin):
        s = requests.Session()
        resp = s.request(
            "GET",
            httpbin("cookies/set?cookie=value"),
            allow_redirects=False,
            headers={"Host": b"httpbin.org"},
        )
        assert resp.cookies.get("cookie") == "value"

    def test_links(self):
        r = requests.Response()
        r.headers = {
            "cache-control": "public, max-age=60, s-maxage=60",
            "connection": "keep-alive",
            "content-encoding": "gzip",
            "content-type": "application/json; charset=utf-8",
            "date": "Sat, 26 Jan 2013 16:47:56 GMT",
            "etag": '"6ff6a73c0e446c1f61614769e3ceb778"',
            "last-modified": "Sat, 26 Jan 2013 16:22:39 GMT",
            "link": (
                "<https://api.github.com/users/kennethreitz/repos?"
                'page=2&per_page=10>; rel="next", <https://api.github.'
                "com/users/kennethreitz/repos?page=7&per_page=10>; "
                ' rel="last"'
            ),
            "server": "GitHub.com",
            "status": "200 OK",
            "vary": "Accept",
            "x-content-type-options": "nosniff",
            "x-github-media-type": "github.beta",
            "x-ratelimit-limit": "60",
            "x-ratelimit-remaining": "57",
        }
        assert r.links["next"]["rel"] == "next"

    def test_cookie_parameters(self):
        key = "some_cookie"
        value = "some_value"
        secure = True
        domain = "test.com"
        rest = {"HttpOnly": True}

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value, secure=secure, domain=domain, rest=rest)

        assert len(jar) == 1
        assert "some_cookie" in jar

        cookie = list(jar)[0]
        assert cookie.secure == secure
        assert cookie.domain == domain
        assert cookie._rest["HttpOnly"] == rest["HttpOnly"]

    def test_cookie_as_dict_keeps_len(self):
        key = "some_cookie"
        value = "some_value"

        key1 = "some_cookie1"
        value1 = "some_value1"

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value)
        jar.set(key1, value1)

        d1 = dict(jar)
        d2 = dict(jar.iteritems())
        d3 = dict(jar.items())

        assert len(jar) == 2
        assert len(d1) == 2
        assert len(d2) == 2
        assert len(d3) == 2

    def test_cookie_as_dict_keeps_items(self):
        key = "some_cookie"
        value = "some_value"

        key1 = "some_cookie1"
        value1 = "some_value1"

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value)
        jar.set(key1, value1)

        d1 = dict(jar)
        d2 = dict(jar.iteritems())
        d3 = dict(jar.items())

        assert d1["some_cookie"] == "some_value"
        assert d2["some_cookie"] == "some_value"
        assert d3["some_cookie1"] == "some_value1"

    def test_cookie_as_dict_keys(self):
        key = "some_cookie"
        value = "some_value"

        key1 = "some_cookie1"
        value1 = "some_value1"

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value)
        jar.set(key1, value1)

        keys = jar.keys()
        assert keys == list(keys)
        # make sure one can use keys multiple times
        assert list(keys) == list(keys)

    def test_cookie_as_dict_values(self):
        key = "some_cookie"
        value = "some_value"

        key1 = "some_cookie1"
        value1 = "some_value1"

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value)
        jar.set(key1, value1)

        values = jar.values()
        assert values == list(values)
        # make sure one can use values multiple times
        assert list(values) == list(values)

    def test_cookie_as_dict_items(self):
        key = "some_cookie"
        value = "some_value"

        key1 = "some_cookie1"
        value1 = "some_value1"

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value)
        jar.set(key1, value1)

        items = jar.items()
        assert items == list(items)
        # make sure one can use items multiple times
        assert list(items) == list(items)

    def test_cookie_duplicate_names_different_domains(self):
        key = "some_cookie"
        value = "some_value"
        domain1 = "test1.com"
        domain2 = "test2.com"

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value, domain=domain1)
        jar.set(key, value, domain=domain2)
        assert key in jar
        items = jar.items()
        assert len(items) == 2

        # Verify that CookieConflictError is raised if domain is not specified
        with pytest.raises(requests.cookies.CookieConflictError):
            jar.get(key)

        # Verify that CookieConflictError is not raised if domain is specified
        cookie = jar.get(key, domain=domain1)
        assert cookie == value

    def test_cookie_duplicate_names_raises_cookie_conflict_error(self):
        key = "some_cookie"
        value = "some_value"
        path = "some_path"

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value, path=path)
        jar.set(key, value)
        with pytest.raises(requests.cookies.CookieConflictError):
            jar.get(key)

    def test_cookie_policy_copy(self):
        class MyCookiePolicy(cookielib.DefaultCookiePolicy):
            pass

        jar = requests.cookies.RequestsCookieJar()
        jar.set_policy(MyCookiePolicy())
        assert isinstance(jar.copy().get_policy(), MyCookiePolicy)

    def test_time_elapsed_blank(self, httpbin):
        r = requests.get(httpbin("get"))
        td = r.elapsed
        total_seconds = (
            td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6
        ) / 10**6
        assert total_seconds > 0.0

    def test_empty_response_has_content_none(self):
        r = requests.Response()
        assert r.content is None

    def test_response_is_iterable(self):
        r = requests.Response()
        io = StringIO.StringIO("abc")
        read_ = io.read

        def read_mock(amt, decode_content=None):
            return read_(amt)

        setattr(io, "read", read_mock)
        r.raw = io
        assert next(iter(r))
        io.close()

    def test_response_decode_unicode(self):
        """When called with decode_unicode, Response.iter_content should always
        return unicode.
        """
        r = requests.Response()
        r._content_consumed = True
        r._content = b"the content"
        r.encoding = "ascii"

        chunks = r.iter_content(decode_unicode=True)
        assert all(isinstance(chunk, str) for chunk in chunks)

        # also for streaming
        r = requests.Response()
        r.raw = io.BytesIO(b"the content")
        r.encoding = "ascii"
        chunks = r.iter_content(decode_unicode=True)
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_response_reason_unicode(self):
        # check for unicode HTTP status
        r = requests.Response()
        r.url = "unicode URL"
        r.reason = "Komponenttia ei löydy".encode()
        r.status_code = 404
        r.encoding = None
        assert not r.ok  # old behaviour - crashes here

    def test_response_reason_unicode_fallback(self):
        # check raise_status falls back to ISO-8859-1
        r = requests.Response()
        r.url = "some url"
        reason = "Komponenttia ei löydy"
        r.reason = reason.encode("latin-1")
        r.status_code = 500
        r.encoding = None
        with pytest.raises(requests.exceptions.HTTPError) as e:
            r.raise_for_status()
        assert reason in e.value.args[0]

    def test_response_chunk_size_type(self):
        """Ensure that chunk_size is passed as None or an integer, otherwise
        raise a TypeError.
        """
        r = requests.Response()
        r.raw = io.BytesIO(b"the content")
        chunks = r.iter_content(1)
        assert all(len(chunk) == 1 for chunk in chunks)

        r = requests.Response()
        r.raw = io.BytesIO(b"the content")
        chunks = r.iter_content(None)
        assert list(chunks) == [b"the content"]

        r = requests.Response()
        r.raw = io.BytesIO(b"the content")
        with pytest.raises(TypeError):
            chunks = r.iter_content("1024")

    @pytest.mark.parametrize(
        "exception, args, expected",
        (
            (urllib3.exceptions.ProtocolError, tuple(), ChunkedEncodingError),
            (urllib3.exceptions.DecodeError, tuple(), ContentDecodingError),
            (urllib3.exceptions.ReadTimeoutError, (None, "", ""), ConnectionError),
            (urllib3.exceptions.SSLError, tuple(), RequestsSSLError),
        ),
    )
    def test_iter_content_wraps_exceptions(self, httpbin, exception, args, expected):
        r = requests.Response()
        r.raw = mock.Mock()
        # ReadTimeoutError can't be initialized by mock
        # so we'll manually create the instance with args
        r.raw.stream.side_effect = exception(*args)

        with pytest.raises(expected):
            next(r.iter_content(1024))

    def test_request_and_response_are_pickleable(self, httpbin):
        r = requests.get(httpbin("get"))

        # verify we can pickle the original request
        assert pickle.loads(pickle.dumps(r.request))

        # verify we can pickle the response and that we have access to
        # the original request.
        pr = pickle.loads(pickle.dumps(r))
        assert r.request.url == pr.request.url
        assert r.request.headers == pr.request.headers

    def test_prepared_request_is_pickleable(self, httpbin):
        p = requests.Request("GET", httpbin("get")).prepare()

        # Verify PreparedRequest can be pickled and unpickled
        r = pickle.loads(pickle.dumps(p))
        assert r.url == p.url
        assert r.headers == p.headers
        assert r.body == p.body

        # Verify unpickled PreparedRequest sends properly
        s = requests.Session()
        resp = s.send(r)
        assert resp.status_code == 200

    def test_prepared_request_with_file_is_pickleable(self, httpbin):
        with open(__file__, "rb") as f:
            r = requests.Request("POST", httpbin("post"), files={"file": f})
            p = r.prepare()

        # Verify PreparedRequest can be pickled and unpickled
        r = pickle.loads(pickle.dumps(p))
        assert r.url == p.url
        assert r.headers == p.headers
        assert r.body == p.body

        # Verify unpickled PreparedRequest sends properly
        s = requests.Session()
        resp = s.send(r)
        assert resp.status_code == 200

    def test_prepared_request_with_hook_is_pickleable(self, httpbin):
        r = requests.Request("GET", httpbin("get"), hooks=default_hooks())
        p = r.prepare()

        # Verify PreparedRequest can be pickled
        r = pickle.loads(pickle.dumps(p))
        assert r.url == p.url
        assert r.headers == p.headers
        assert r.body == p.body
        assert r.hooks == p.hooks

        # Verify unpickled PreparedRequest sends properly
        s = requests.Session()
        resp = s.send(r)
        assert resp.status_code == 200

    def test_cannot_send_unprepared_requests(self, httpbin):
        r = requests.Request(url=httpbin())
        with pytest.raises(ValueError):
            requests.Session().send(r)

    def test_http_error(self):
        error = requests.exceptions.HTTPError()
        assert not error.response
        response = requests.Response()
        error = requests.exceptions.HTTPError(response=response)
        assert error.response == response
        error = requests.exceptions.HTTPError("message", response=response)
        assert str(error) == "message"
        assert error.response == response

    def test_session_pickling(self, httpbin):
        r = requests.Request("GET", httpbin("get"))
        s = requests.Session()

        s = pickle.loads(pickle.dumps(s))
        s.proxies = getproxies()

        r = s.send(r.prepare())
        assert r.status_code == 200

    def test_fixes_1329(self, httpbin):
        """Ensure that header updates are done case-insensitively."""
        s = requests.Session()
        s.headers.update({"ACCEPT": "BOGUS"})
        s.headers.update({"accept": "application/json"})
        r = s.get(httpbin("get"))
        headers = r.request.headers
        assert headers["accept"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert headers["ACCEPT"] == "application/json"

    def test_uppercase_scheme_redirect(self, httpbin):
        parts = urlparse(httpbin("html"))
        url = "HTTP://" + parts.netloc + parts.path
        r = requests.get(httpbin("redirect-to"), params={"url": url})
        assert r.status_code == 200
        assert r.url.lower() == url.lower()

    def test_transport_adapter_ordering(self):
        s = requests.Session()
        order = ["https://", "http://"]
        assert order == list(s.adapters)
        s.mount("http://git", HTTPAdapter())
        s.mount("http://github", HTTPAdapter())
        s.mount("http://github.com", HTTPAdapter())
        s.mount("http://github.com/about/", HTTPAdapter())
        order = [
            "http://github.com/about/",
            "http://github.com",
            "http://github",
            "http://git",
            "https://",
            "http://",
        ]
        assert order == list(s.adapters)
        s.mount("http://gittip", HTTPAdapter())
        s.mount("http://gittip.com", HTTPAdapter())
        s.mount("http://gittip.com/about/", HTTPAdapter())
        order = [
            "http://github.com/about/",
            "http://gittip.com/about/",
            "http://github.com",
            "http://gittip.com",
            "http://github",
            "http://gittip",
            "http://git",
            "https://",
            "http://",
        ]
        assert order == list(s.adapters)
        s2 = requests.Session()
        s2.adapters = {"http://": HTTPAdapter()}
        s2.mount("https://", HTTPAdapter())
        assert "http://" in s2.adapters
        assert "https://" in s2.adapters

    def test_session_get_adapter_prefix_matching(self):
        prefix = "https://example.com"
        more_specific_prefix = prefix + "/some/path"

        url_matching_only_prefix = prefix + "/another/path"
        url_matching_more_specific_prefix = more_specific_prefix + "/longer/path"
        url_not_matching_prefix = "https://another.example.com/"

        s = requests.Session()
        prefix_adapter = HTTPAdapter()
        more_specific_prefix_adapter = HTTPAdapter()
        s.mount(prefix, prefix_adapter)
        s.mount(more_specific_prefix, more_specific_prefix_adapter)

        assert s.get_adapter(url_matching_only_prefix) is prefix_adapter
        assert (
            s.get_adapter(url_matching_more_specific_prefix)
            is more_specific_prefix_adapter
        )
        assert s.get_adapter(url_not_matching_prefix) not in (
            prefix_adapter,
            more_specific_prefix_adapter,
        )

    def test_session_get_adapter_prefix_matching_mixed_case(self):
        mixed_case_prefix = "hTtPs://eXamPle.CoM/MixEd_CAse_PREfix"
        url_matching_prefix = mixed_case_prefix + "/full_url"

        s = requests.Session()
        my_adapter = HTTPAdapter()
        s.mount(mixed_case_prefix, my_adapter)

        assert s.get_adapter(url_matching_prefix) is my_adapter

    def test_session_get_adapter_prefix_matching_is_case_insensitive(self):
        mixed_case_prefix = "hTtPs://eXamPle.CoM/MixEd_CAse_PREfix"
        url_matching_prefix_with_different_case = (
            "HtTpS://exaMPLe.cOm/MiXeD_caSE_preFIX/another_url"
        )

        s = requests.Session()
        my_adapter = HTTPAdapter()
        s.mount(mixed_case_prefix, my_adapter)

        assert s.get_adapter(url_matching_prefix_with_different_case) is my_adapter

    def test_session_get_adapter_prefix_with_trailing_slash(self):
        # from issue #6935
        prefix = "https://example.com/"  # trailing slash
        url_matching_prefix = "https://example.com/some/path"
        url_not_matching_prefix = "https://example.com.other.com/some/path"

        s = requests.Session()
        adapter = HTTPAdapter()
        s.mount(prefix, adapter)

        assert s.get_adapter(url_matching_prefix) is adapter
        assert s.get_adapter(url_not_matching_prefix) is not adapter

    def test_session_get_adapter_prefix_without_trailing_slash(self):
        # from issue #6935
        prefix = "https://example.com"  # no trailing slash
        url_matching_prefix = "https://example.com/some/path"
        url_extended_hostname = "https://example.com.other.com/some/path"

        s = requests.Session()
        adapter = HTTPAdapter()
        s.mount(prefix, adapter)

        assert s.get_adapter(url_matching_prefix) is adapter
        assert s.get_adapter(url_extended_hostname) is adapter

    def test_header_remove_is_case_insensitive(self, httpbin):
        # From issue #1321
        s = requests.Session()
        s.headers["foo"] = "bar"
        r = s.get(httpbin("get"), headers={"FOO": None})
        assert "foo" not in r.request.headers

    def test_params_are_merged_case_sensitive(self, httpbin):
        s = requests.Session()
        s.params["foo"] = "bar"
        r = s.get(httpbin("get"), params={"FOO": "bar"})
        assert r.json()["args"] == {"foo": "bar", "FOO": "bar"}

    def test_long_authinfo_in_url(self):
        url = "http://{}:{}@{}:9000/path?query#frag".format(
            "E8A3BE87-9E3F-4620-8858-95478E385B5B",
            "EA770032-DA4D-4D84-8CE9-29C6D910BF1E",
            "exactly-------------sixty-----------three------------characters",
        )
        r = requests.Request("GET", url).prepare()
        assert r.url == url

    def test_header_keys_are_native(self, httpbin):
        headers = {"unicode": "blah", b"byte": "blah"}
        r = requests.Request("GET", httpbin("get"), headers=headers)
        p = r.prepare()

        # This is testing that they are builtin strings. A bit weird, but there
        # we go.
        assert "unicode" in p.headers.keys()
        assert "byte" in p.headers.keys()

    def test_header_validation(self, httpbin):
        """Ensure prepare_headers regex isn't flagging valid header contents."""
        valid_headers = {
            "foo": "bar baz qux",
            "bar": b"fbbq",
            "baz": "",
            "qux": "1",
        }
        r = requests.get(httpbin("get"), headers=valid_headers)
        for key in valid_headers.keys():
            assert valid_headers[key] == r.request.headers[key]

    @pytest.mark.parametrize(
        "invalid_header, key",
        (
            ({"foo": 3}, "foo"),
            ({"bar": {"foo": "bar"}}, "bar"),
            ({"baz": ["foo", "bar"]}, "baz"),
        ),
    )
    def test_header_value_not_str(self, httpbin, invalid_header, key):
        """Ensure the header value is of type string or bytes as
        per discussion in GH issue #3386
        """
        with pytest.raises(InvalidHeader) as excinfo:
            requests.get(httpbin("get"), headers=invalid_header)
        assert key in str(excinfo.value)

    @pytest.mark.parametrize(
        "invalid_header",
        (
            {"foo": "bar\r\nbaz: qux"},
            {"foo": "bar\n\rbaz: qux"},
            {"foo": "bar\nbaz: qux"},
            {"foo": "bar\rbaz: qux"},
            {"fo\ro": "bar"},
            {"fo\r\no": "bar"},
            {"fo\n\ro": "bar"},
            {"fo\no": "bar"},
        ),
    )
    def test_header_no_return_chars(self, httpbin, invalid_header):
        """Ensure that a header containing return character sequences raise an
        exception. Otherwise, multiple headers are created from single string.
        """
        with pytest.raises(InvalidHeader):
            requests.get(httpbin("get"), headers=invalid_header)

    @pytest.mark.parametrize(
        "invalid_header",
        (
            {" foo": "bar"},
            {"\tfoo": "bar"},
            {"    foo": "bar"},
            {"foo": " bar"},
            {"foo": "    bar"},
            {"foo": "\tbar"},
            {" ": "bar"},
        ),
    )
    def test_header_no_leading_space(self, httpbin, invalid_header):
        """Ensure headers containing leading whitespace raise
        InvalidHeader Error before sending.
        """
        with pytest.raises(InvalidHeader):
            requests.get(httpbin("get"), headers=invalid_header)

    def test_header_with_subclass_types(self, httpbin):
        """If the subclasses does not behave *exactly* like
        the base bytes/str classes, this is not supported.
        This test is for backwards compatibility.
        """

        class MyString(str):
            pass

        class MyBytes(bytes):
            pass

        r_str = requests.get(httpbin("get"), headers={MyString("x-custom"): "myheader"})
        assert r_str.request.headers["x-custom"] == "myheader"

        r_bytes = requests.get(
            httpbin("get"), headers={MyBytes(b"x-custom"): b"myheader"}
        )
        assert r_bytes.request.headers["x-custom"] == b"myheader"

        r_mixed = requests.get(
            httpbin("get"), headers={MyString("x-custom"): MyBytes(b"myheader")}
        )
        assert r_mixed.request.headers["x-custom"] == b"myheader"

    @pytest.mark.parametrize("files", ("foo", b"foo", bytearray(b"foo")))
    def test_can_send_objects_with_files(self, httpbin, files):
        data = {"a": "this is a string"}
        files = {"b": files}
        r = requests.Request("POST", httpbin("post"), data=data, files=files)
        p = r.prepare()
        assert "multipart/form-data" in p.headers["Content-Type"]

    def test_can_send_file_object_with_non_string_filename(self, httpbin):
        f = io.BytesIO()
        f.name = 2
        r = requests.Request("POST", httpbin("post"), files={"f": f})
        p = r.prepare()

        assert "multipart/form-data" in p.headers["Content-Type"]

    def test_autoset_header_values_are_native(self, httpbin):
        data = "this is a string"
        length = "16"
        req = requests.Request("POST", httpbin("post"), data=data)
        p = req.prepare()

        assert p.headers["Content-Length"] == length

    def test_nonhttp_schemes_dont_check_URLs(self):
        test_urls = (
            "data:image/gif;base64,R0lGODlhAQABAHAAACH5BAUAAAAALAAAAAABAAEAAAICRAEAOw==",
            "file:///etc/passwd",
            "magnet:?xt=urn:btih:be08f00302bc2d1d3cfa3af02024fa647a271431",
        )
        for test_url in test_urls:
            req = requests.Request("GET", test_url)
            preq = req.prepare()
            assert test_url == preq.url

    def test_auth_is_stripped_on_http_downgrade(
        self, httpbin, httpbin_secure, httpbin_ca_bundle
    ):
        r = requests.get(
            httpbin_secure("redirect-to"),
            params={"url": httpbin("get")},
            auth=("user", "pass"),
            verify=httpbin_ca_bundle,
        )
        assert r.history[0].request.headers["Authorization"]
        assert "Authorization" not in r.request.headers

    def test_auth_is_retained_for_redirect_on_host(self, httpbin):
        r = requests.get(httpbin("redirect/1"), auth=("user", "pass"))
        h1 = r.history[0].request.headers["Authorization"]
        h2 = r.request.headers["Authorization"]

        assert h1 == h2

    def test_should_strip_auth_host_change(self):
        s = requests.Session()
        assert s.should_strip_auth(
            "http://example.com/foo", "http://another.example.com/"
        )

    def test_should_strip_auth_http_downgrade(self):
        s = requests.Session()
        assert s.should_strip_auth("https://example.com/foo", "http://example.com/bar")

    def test_should_strip_auth_https_upgrade(self):
        s = requests.Session()
        assert not s.should_strip_auth(
            "http://example.com/foo", "https://example.com/bar"
        )
        assert not s.should_strip_auth(
            "http://example.com:80/foo", "https://example.com/bar"
        )
        assert not s.should_strip_auth(
            "http://example.com/foo", "https://example.com:443/bar"
        )
        # Non-standard ports should trigger stripping
        assert s.should_strip_auth(
            "http://example.com:8080/foo", "https://example.com/bar"
        )
        assert s.should_strip_auth(
            "http://example.com/foo", "https://example.com:8443/bar"
        )

    def test_should_strip_auth_port_change(self):
        s = requests.Session()
        assert s.should_strip_auth(
            "http://example.com:1234/foo", "https://example.com:4321/bar"
        )

    @pytest.mark.parametrize(
        "old_uri, new_uri",
        (
            ("https://example.com:443/foo", "https://example.com/bar"),
            ("http://example.com:80/foo", "http://example.com/bar"),
            ("https://example.com/foo", "https://example.com:443/bar"),
            ("http://example.com/foo", "http://example.com:80/bar"),
        ),
    )
    def test_should_strip_auth_default_port(self, old_uri, new_uri):
        s = requests.Session()
        assert not s.should_strip_auth(old_uri, new_uri)

    def test_manual_redirect_with_partial_body_read(self, httpbin):
        s = requests.Session()
        r1 = s.get(httpbin("redirect/2"), allow_redirects=False, stream=True)
        assert r1.is_redirect
        rg = s.resolve_redirects(r1, r1.request, stream=True)

        # read only the first eight bytes of the response body,
        # then follow the redirect
        r1.iter_content(8)
        r2 = next(rg)
        assert r2.is_redirect

        # read all of the response via iter_content,
        # then follow the redirect
        for _ in r2.iter_content():
            pass
        r3 = next(rg)
        assert not r3.is_redirect

    def test_prepare_body_position_non_stream(self):
        data = b"the data"
        prep = requests.Request("GET", "http://example.com", data=data).prepare()
        assert prep._body_position is None

    def test_rewind_body(self):
        data = io.BytesIO(b"the data")
        prep = requests.Request("GET", "http://example.com", data=data).prepare()
        assert prep._body_position == 0
        assert prep.body.read() == b"the data"

        # the data has all been read
        assert prep.body.read() == b""

        # rewind it back
        requests.utils.rewind_body(prep)
        assert prep.body.read() == b"the data"

    def test_rewind_partially_read_body(self):
        data = io.BytesIO(b"the data")
        data.read(4)  # read some data
        prep = requests.Request("GET", "http://example.com", data=data).prepare()
        assert prep._body_position == 4
        assert prep.body.read() == b"data"

        # the data has all been read
        assert prep.body.read() == b""

        # rewind it back
        requests.utils.rewind_body(prep)
        assert prep.body.read() == b"data"

    def test_rewind_body_no_seek(self):
        class BadFileObj:
            def __init__(self, data):
                self.data = data

            def tell(self):
                return 0

            def __iter__(self):
                return

        data = BadFileObj("the data")
        prep = requests.Request("GET", "http://example.com", data=data).prepare()
        assert prep._body_position == 0

        with pytest.raises(UnrewindableBodyError) as e:
            requests.utils.rewind_body(prep)

        assert "Unable to rewind request body" in str(e)

    def test_rewind_body_failed_seek(self):
        class BadFileObj:
            def __init__(self, data):
                self.data = data

            def tell(self):
                return 0

            def seek(self, pos, whence=0):
                raise OSError()

            def __iter__(self):
                return

        data = BadFileObj("the data")
        prep = requests.Request("GET", "http://example.com", data=data).prepare()
        assert prep._body_position == 0

        with pytest.raises(UnrewindableBodyError) as e:
            requests.utils.rewind_body(prep)

        assert "error occurred when rewinding request body" in str(e)

    def test_rewind_body_failed_tell(self):
        class BadFileObj:
            def __init__(self, data):
                self.data = data

            def tell(self):
                raise OSError()

            def __iter__(self):
                return

        data = BadFileObj("the data")
        prep = requests.Request("GET", "http://example.com", data=data).prepare()
        assert prep._body_position is not None

        with pytest.raises(UnrewindableBodyError) as e:
            requests.utils.rewind_body(prep)

        assert "Unable to rewind request body" in str(e)

    def _patch_adapter_gzipped_redirect(self, session, url):
        adapter = session.get_adapter(url=url)
        org_build_response = adapter.build_response
        self._patched_response = False

        def build_response(*args, **kwargs):
            resp = org_build_response(*args, **kwargs)
            if not self._patched_response:
                resp.raw.headers["content-encoding"] = "gzip"
                self._patched_response = True
            return resp

        adapter.build_response = build_response

    def test_redirect_with_wrong_gzipped_header(self, httpbin):
        s = requests.Session()
        url = httpbin("redirect/1")
        self._patch_adapter_gzipped_redirect(s, url)
        s.get(url)

    @pytest.mark.parametrize(
        "username, password, auth_str",
        (
            ("test", "test", "Basic dGVzdDp0ZXN0"),
            (
                "имя".encode(),
                "пароль".encode(),
                "Basic 0LjQvNGPOtC/0LDRgNC+0LvRjA==",
            ),
        ),
    )
    def test_basic_auth_str_is_always_native(self, username, password, auth_str):
        s = _basic_auth_str(username, password)
        assert isinstance(s, builtin_str)
        assert s == auth_str

    def test_requests_history_is_saved(self, httpbin):
        r = requests.get(httpbin("redirect/5"))
        total = r.history[-1].history
        i = 0
        for item in r.history:
            assert item.history == total[0:i]
            i += 1

    def test_json_param_post_content_type_works(self, httpbin):
        r = requests.post(httpbin("post"), json={"life": 42})
        assert r.status_code == 200
        assert "application/json" in r.request.headers["Content-Type"]
        assert {"life": 42} == r.json()["json"]

    def test_json_param_post_should_not_override_data_param(self, httpbin):
        r = requests.Request(
            method="POST",
            url=httpbin("post"),
            data={"stuff": "elixr"},
            json={"music": "flute"},
        )
        prep = r.prepare()
        assert "stuff=elixr" == prep.body

    def test_response_iter_lines(self, httpbin):
        r = requests.get(httpbin("stream/4"), stream=True)
        assert r.status_code == 200

        it = r.iter_lines()
        next(it)
        assert len(list(it)) == 3

    def test_response_context_manager(self, httpbin):
        with requests.get(httpbin("stream/4"), stream=True) as response:
            assert isinstance(response, requests.Response)

        assert response.raw.closed

    def test_unconsumed_session_response_closes_connection(self, httpbin):
        s = requests.session()

        with contextlib.closing(s.get(httpbin("stream/4"), stream=True)) as response:
            pass

        assert response._content_consumed is False
        assert response.raw.closed

    @pytest.mark.xfail
    def test_response_iter_lines_reentrant(self, httpbin):
        """Response.iter_lines() is not reentrant safe"""
        r = requests.get(httpbin("stream/4"), stream=True)
        assert r.status_code == 200

        next(r.iter_lines())
        assert len(list(r.iter_lines())) == 3

    def test_session_close_proxy_clear(self):
        proxies = {
            "one": mock.Mock(),
            "two": mock.Mock(),
        }
        session = requests.Session()
        with mock.patch.dict(session.adapters["http://"].proxy_manager, proxies):
            session.close()
            proxies["one"].clear.assert_called_once_with()
            proxies["two"].clear.assert_called_once_with()

    def test_proxy_auth(self):
        adapter = HTTPAdapter()
        headers = adapter.proxy_headers("http://user:pass@httpbin.org")
        assert headers == {"Proxy-Authorization": "Basic dXNlcjpwYXNz"}

    def test_proxy_auth_empty_pass(self):
        adapter = HTTPAdapter()
        headers = adapter.proxy_headers("http://user:@httpbin.org")
        assert headers == {"Proxy-Authorization": "Basic dXNlcjo="}

    def test_response_json_when_content_is_None(self, httpbin):
        r = requests.get(httpbin("/status/204"))
        # Make sure r.content is None
        r.status_code = 0
        r._content = False
        r._content_consumed = False

        assert r.content is None
        with pytest.raises(ValueError):
            r.json()

    def test_response_without_release_conn(self):
        """Test `close` call for non-urllib3-like raw objects.
        Should work when `release_conn` attr doesn't exist on `response.raw`.
        """
        resp = requests.Response()
        resp.raw = StringIO.StringIO("test")
        assert not resp.raw.closed
        resp.close()
        assert resp.raw.closed

    def test_empty_stream_with_auth_does_not_set_content_length_header(self, httpbin):
        """Ensure that a byte stream with size 0 will not set both a Content-Length
        and Transfer-Encoding header.
        """
        auth = ("user", "pass")
        url = httpbin("post")
        file_obj = io.BytesIO(b"")
        r = requests.Request("POST", url, auth=auth, data=file_obj)
        prepared_request = r.prepare()
        assert "Transfer-Encoding" in prepared_request.headers
        assert "Content-Length" not in prepared_request.headers

    def test_stream_with_auth_does_not_set_transfer_encoding_header(self, httpbin):
        """Ensure that a byte stream with size > 0 will not set both a Content-Length
        and Transfer-Encoding header.
        """
        auth = ("user", "pass")
        url = httpbin("post")
        file_obj = io.BytesIO(b"test data")
        r = requests.Request("POST", url, auth=auth, data=file_obj)
        prepared_request = r.prepare()
        assert "Transfer-Encoding" not in prepared_request.headers
        assert "Content-Length" in prepared_request.headers

    def test_chunked_upload_does_not_set_content_length_header(self, httpbin):
        """Ensure that requests with a generator body stream using
        Transfer-Encoding: chunked, not a Content-Length header.
        """
        data = (i for i in [b"a", b"b", b"c"])
        url = httpbin("post")
        r = requests.Request("POST", url, data=data)
        prepared_request = r.prepare()
        assert "Transfer-Encoding" in prepared_request.headers
        assert "Content-Length" not in prepared_request.headers

    def test_custom_redirect_mixin(self, httpbin):
        """Tests a custom mixin to overwrite ``get_redirect_target``.

        Ensures a subclassed ``requests.Session`` can handle a certain type of
        malformed redirect responses.

        1. original request receives a proper response: 302 redirect
        2. following the redirect, a malformed response is given:
            status code = HTTP 200
            location = alternate url
        3. the custom session catches the edge case and follows the redirect
        """
        url_final = httpbin("html")
        querystring_malformed = urlencode({"location": url_final})
        url_redirect_malformed = httpbin("response-headers?%s" % querystring_malformed)
        querystring_redirect = urlencode({"url": url_redirect_malformed})
        url_redirect = httpbin("redirect-to?%s" % querystring_redirect)
        urls_test = [
            url_redirect,
            url_redirect_malformed,
            url_final,
        ]

        class CustomRedirectSession(requests.Session):
            def get_redirect_target(self, resp):
                # default behavior
                if resp.is_redirect:
                    return resp.headers["location"]
                # edge case - check to see if 'location' is in headers anyways
                location = resp.headers.get("location")
                if location and (location != resp.url):
                    return location
                return None

        session = CustomRedirectSession()
        r = session.get(urls_test[0])
        assert len(r.history) == 2
        assert r.status_code == 200
        assert r.history[0].status_code == 302
        assert r.history[0].is_redirect
        assert r.history[1].status_code == 200
        assert not r.history[1].is_redirect
        assert r.url == urls_test[2]


class TestCaseInsensitiveDict:
    @pytest.mark.parametrize(
        "cid",
        (
            CaseInsensitiveDict({"Foo": "foo", "BAr": "bar"}),
            CaseInsensitiveDict([("Foo", "foo"), ("BAr", "bar")]),
            CaseInsensitiveDict(FOO="foo", BAr="bar"),
        ),
    )
    def test_init(self, cid):
        assert len(cid) == 2
        assert "foo" in cid
        assert "bar" in cid

    def test_docstring_example(self):
        cid = CaseInsensitiveDict()
        cid["Accept"] = "application/json"
        assert cid["aCCEPT"] == "application/json"
        assert list(cid) == ["Accept"]

    def test_len(self):
        cid = CaseInsensitiveDict({"a": "a", "b": "b"})
        cid["A"] = "a"
        assert len(cid) == 2

    def test_getitem(self):
        cid = CaseInsensitiveDict({"Spam": "blueval"})
        assert cid["spam"] == "blueval"
        assert cid["SPAM"] == "blueval"

    def test_fixes_649(self):
        """__setitem__ should behave case-insensitively."""
        cid = CaseInsensitiveDict()
        cid["spam"] = "oneval"
        cid["Spam"] = "twoval"
        cid["sPAM"] = "redval"
        cid["SPAM"] = "blueval"
        assert cid["spam"] == "blueval"
        assert cid["SPAM"] == "blueval"
        assert list(cid.keys()) == ["SPAM"]

    def test_delitem(self):
        cid = CaseInsensitiveDict()
        cid["Spam"] = "someval"
        del cid["sPam"]
        assert "spam" not in cid
        assert len(cid) == 0

    def test_contains(self):
        cid = CaseInsensitiveDict()
        cid["Spam"] = "someval"
        assert "Spam" in cid
        assert "spam" in cid
        assert "SPAM" in cid
        assert "sPam" in cid
        assert "notspam" not in cid

    def test_get(self):
        cid = CaseInsensitiveDict()
        cid["spam"] = "oneval"
        cid["SPAM"] = "blueval"
        assert cid.get("spam") == "blueval"
        assert cid.get("SPAM") == "blueval"
        assert cid.get("sPam") == "blueval"
        assert cid.get("notspam", "default") == "default"

    def test_update(self):
        cid = CaseInsensitiveDict()
        cid["spam"] = "blueval"
        cid.update({"sPam": "notblueval"})
        assert cid["spam"] == "notblueval"
        cid = CaseInsensitiveDict({"Foo": "foo", "BAr": "bar"})
        cid.update({"fOO": "anotherfoo", "bAR": "anotherbar"})
        assert len(cid) == 2
        assert cid["foo"] == "anotherfoo"
        assert cid["bar"] == "anotherbar"

    def test_update_retains_unchanged(self):
        cid = CaseInsensitiveDict({"foo": "foo", "bar": "bar"})
        cid.update({"foo": "newfoo"})
        assert cid["bar"] == "bar"

    def test_iter(self):
        cid = CaseInsensitiveDict({"Spam": "spam", "Eggs": "eggs"})
        keys = frozenset(["Spam", "Eggs"])
        assert frozenset(iter(cid)) == keys

    def test_equality(self):
        cid = CaseInsensitiveDict({"SPAM": "blueval", "Eggs": "redval"})
        othercid = CaseInsensitiveDict({"spam": "blueval", "eggs": "redval"})
        assert cid == othercid
        del othercid["spam"]
        assert cid != othercid
        assert cid == {"spam": "blueval", "eggs": "redval"}
        assert cid != object()

    def test_setdefault(self):
        cid = CaseInsensitiveDict({"Spam": "blueval"})
        assert cid.setdefault("spam", "notblueval") == "blueval"
        assert cid.setdefault("notspam", "notblueval") == "notblueval"

    def test_lower_items(self):
        cid = CaseInsensitiveDict(
            {
                "Accept": "application/json",
                "user-Agent": "requests",
            }
        )
        keyset = frozenset(lowerkey for lowerkey, v in cid.lower_items())
        lowerkeyset = frozenset(["accept", "user-agent"])
        assert keyset == lowerkeyset

    def test_preserve_key_case(self):
        cid = CaseInsensitiveDict(
            {
                "Accept": "application/json",
                "user-Agent": "requests",
            }
        )
        keyset = frozenset(["Accept", "user-Agent"])
        assert frozenset(i[0] for i in cid.items()) == keyset
        assert frozenset(cid.keys()) == keyset
        assert frozenset(cid) == keyset

    def test_preserve_last_key_case(self):
        cid = CaseInsensitiveDict(
            {
                "Accept": "application/json",
                "user-Agent": "requests",
            }
        )
        cid.update({"ACCEPT": "application/json"})
        cid["USER-AGENT"] = "requests"
        keyset = frozenset(["ACCEPT", "USER-AGENT"])
        assert frozenset(i[0] for i in cid.items()) == keyset
        assert frozenset(cid.keys()) == keyset
        assert frozenset(cid) == keyset

    def test_copy(self):
        cid = CaseInsensitiveDict(
            {
                "Accept": "application/json",
                "user-Agent": "requests",
            }
        )
        cid_copy = cid.copy()
        assert cid == cid_copy
        cid["changed"] = True
        assert cid != cid_copy


class TestMorselToCookieExpires:
    """Tests for morsel_to_cookie when morsel contains expires."""

    def test_expires_valid_str(self):
        """Test case where we convert expires from string time."""

        morsel = Morsel()
        morsel["expires"] = "Thu, 01-Jan-1970 00:00:01 GMT"
        cookie = morsel_to_cookie(morsel)
        assert cookie.expires == 1

    @pytest.mark.parametrize(
        "value, exception",
        (
            (100, TypeError),
            ("woops", ValueError),
        ),
    )
    def test_expires_invalid_int(self, value, exception):
        """Test case where an invalid type is passed for expires."""
        morsel = Morsel()
        morsel["expires"] = value
        with pytest.raises(exception):
            morsel_to_cookie(morsel)

    def test_expires_none(self):
        """Test case where expires is None."""

        morsel = Morsel()
        morsel["expires"] = None
        cookie = morsel_to_cookie(morsel)
        assert cookie.expires is None


class TestMorselToCookieMaxAge:

    """Tests for morsel_to_cookie when morsel contains max-age."""

    def test_max_age_valid_int(self):
        """Test case where a valid max age in seconds is passed."""

        morsel = Morsel()
        morsel["max-age"] = 60
        cookie = morsel_to_cookie(morsel)
        assert isinstance(cookie.expires, int)

    def test_max_age_invalid_str(self):
        """Test case where a invalid max age is passed."""

        morsel = Morsel()
        morsel["max-age"] = "woops"
        with pytest.raises(TypeError):
            morsel_to_cookie(morsel)


class TestTimeout:
    def test_stream_timeout(self, httpbin):
        try:
            requests.get(httpbin("delay/10"), timeout=2.0)
        except requests.exceptions.Timeout as e:
            assert "Read timed out" in e.args[0].args[0]

    @pytest.mark.parametrize(
        "timeout, error_text",
        (
            ((3, 4, 5), "(connect, read)"),
            ("foo", "must be an int, float or None"),
        ),
    )
    def test_invalid_timeout(self, httpbin, timeout, error_text):
        with pytest.raises(ValueError) as e:
            requests.get(httpbin("get"), timeout=timeout)
        assert error_text in str(e)

    @pytest.mark.parametrize("timeout", (None, Urllib3Timeout(connect=None, read=None)))
    def test_none_timeout(self, httpbin, timeout):
        """Check that you can set None as a valid timeout value.

        To actually test this behavior, we'd want to check that setting the
        timeout to None actually lets the request block past the system default
        timeout. However, this would make the test suite unbearably slow.
        Instead we verify that setting the timeout to None does not prevent the
        request from succeeding.
        """
        r = requests.get(httpbin("get"), timeout=timeout)
        assert r.status_code == 200

    @pytest.mark.parametrize(
        "timeout", ((None, 0.1), Urllib3Timeout(connect=None, read=0.1))
    )
    def test_read_timeout(self, httpbin, timeout):
        try:
            requests.get(httpbin("delay/10"), timeout=timeout)
            pytest.fail("The recv() request should time out.")
        except ReadTimeout:
            pass

    @pytest.mark.parametrize(
        "timeout", ((0.1, None), Urllib3Timeout(connect=0.1, read=None))
    )
    def test_connect_timeout(self, timeout):
        try:
            requests.get(TARPIT, timeout=timeout)
            pytest.fail("The connect() request should time out.")
        except ConnectTimeout as e:
            assert isinstance(e, ConnectionError)
            assert isinstance(e, Timeout)

    @pytest.mark.parametrize(
        "timeout", ((0.1, 0.1), Urllib3Timeout(connect=0.1, read=0.1))
    )
    def test_total_timeout_connect(self, timeout):
        try:
            requests.get(TARPIT, timeout=timeout)
            pytest.fail("The connect() request should time out.")
        except ConnectTimeout:
            pass

    def test_encoded_methods(self, httpbin):
        """See: https://github.com/psf/requests/issues/2316"""
        r = requests.request(b"GET", httpbin("get"))
        assert r.ok


SendCall = collections.namedtuple("SendCall", ("args", "kwargs"))


class RedirectSession(SessionRedirectMixin):
    def __init__(self, order_of_redirects):
        self.redirects = order_of_redirects
        self.calls = []
        self.max_redirects = 30
        self.cookies = {}
        self.trust_env = False

    def send(self, *args, **kwargs):
        self.calls.append(SendCall(args, kwargs))
        return self.build_response()

    def build_response(self):
        request = self.calls[-1].args[0]
        r = requests.Response()

        try:
            r.status_code = int(self.redirects.pop(0))
        except IndexError:
            r.status_code = 200

        r.headers = CaseInsensitiveDict({"Location": "/"})
        r.raw = self._build_raw()
        r.request = request
        return r

    def _build_raw(self):
        string = StringIO.StringIO("")
        setattr(string, "release_conn", lambda *args: args)
        return string


def test_json_encodes_as_bytes():
    # urllib3 expects bodies as bytes-like objects
    body = {"key": "value"}
    p = PreparedRequest()
    p.prepare(method="GET", url="https://www.example.com/", json=body)
    assert isinstance(p.body, bytes)


def test_requests_are_updated_each_time(httpbin):
    session = RedirectSession([303, 307])
    prep = requests.Request("POST", httpbin("post")).prepare()
    r0 = session.send(prep)
    assert r0.request.method == "POST"
    assert session.calls[-1] == SendCall((r0.request,), {})
    redirect_generator = session.resolve_redirects(r0, prep)
    default_keyword_args = {
        "stream": False,
        "verify": True,
        "cert": None,
        "timeout": None,
        "allow_redirects": False,
        "proxies": {},
    }
    for response in redirect_generator:
        assert response.request.method == "GET"
        send_call = SendCall((response.request,), default_keyword_args)
        assert session.calls[-1] == send_call


@pytest.mark.parametrize(
    "var,url,proxy",
    [
        ("http_proxy", "http://example.com", "socks5://proxy.com:9876"),
        ("https_proxy", "https://example.com", "socks5://proxy.com:9876"),
        ("all_proxy", "http://example.com", "socks5://proxy.com:9876"),
        ("all_proxy", "https://example.com", "socks5://proxy.com:9876"),
    ],
)
def test_proxy_env_vars_override_default(var, url, proxy):
    session = requests.Session()
    prep = PreparedRequest()
    prep.prepare(method="GET", url=url)

    kwargs = {var: proxy}
    scheme = urlparse(url).scheme
    with override_environ(**kwargs):
        proxies = session.rebuild_proxies(prep, {})
        assert scheme in proxies
        assert proxies[scheme] == proxy


@pytest.mark.parametrize(
    "data",
    (
        (("a", "b"), ("c", "d")),
        (("c", "d"), ("a", "b")),
        (("a", "b"), ("c", "d"), ("e", "f")),
    ),
)
def test_data_argument_accepts_tuples(data):
    """Ensure that the data argument will accept tuples of strings
    and properly encode them.
    """
    p = PreparedRequest()
    p.prepare(
        method="GET", url="http://www.example.com", data=data, hooks=default_hooks()
    )
    assert p.body == urlencode(data)


@pytest.mark.parametrize(
    "kwargs",
    (
        None,
        {
            "method": "GET",
            "url": "http://www.example.com",
            "data": "foo=bar",
            "hooks": default_hooks(),
        },
        {
            "method": "GET",
            "url": "http://www.example.com",
            "data": "foo=bar",
            "hooks": default_hooks(),
            "cookies": {"foo": "bar"},
        },
        {"method": "GET", "url": "http://www.example.com/üniçø∂é"},
    ),
)
def test_prepared_copy(kwargs):
    p = PreparedRequest()
    if kwargs:
        p.prepare(**kwargs)
    copy = p.copy()
    for attr in ("method", "url", "headers", "_cookies", "body", "hooks"):
        assert getattr(p, attr) == getattr(copy, attr)


def test_urllib3_retries(httpbin):
    from urllib3.util import Retry

    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=Retry(total=2, status_forcelist=[500])))

    with pytest.raises(RetryError):
        s.get(httpbin("status/500"))


def test_urllib3_pool_connection_closed(httpbin):
    s = requests.Session()
    s.mount("http://", HTTPAdapter(pool_connections=0, pool_maxsize=0))

    try:
        s.get(httpbin("status/200"))
    except ConnectionError as e:
        assert "Pool is closed." in str(e)


class TestPreparingURLs:
    @pytest.mark.parametrize(
        "url,expected",
        (
            ("http://google.com", "http://google.com/"),
            ("http://ジェーピーニック.jp", "http://xn--hckqz9bzb1cyrb.jp/"),
            ("http://xn--n3h.net/", "http://xn--n3h.net/"),
            ("http://ジェーピーニック.jp".encode(), "http://xn--hckqz9bzb1cyrb.jp/"),
            ("http://straße.de/straße", "http://xn--strae-oqa.de/stra%C3%9Fe"),
            (
                "http://straße.de/straße".encode(),
                "http://xn--strae-oqa.de/stra%C3%9Fe",
            ),
            (
                "http://Königsgäßchen.de/straße",
                "http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe",
            ),
            (
                "http://Königsgäßchen.de/straße".encode(),
                "http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe",
            ),
            (b"http://xn--n3h.net/", "http://xn--n3h.net/"),
            (
                b"http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/",
                "http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/",
            ),
            (
                "http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/",
                "http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/",
            ),
        ),
    )
    def test_preparing_url(self, url, expected):
        def normalize_percent_encode(x):
            # Helper function that normalizes equivalent
            # percent-encoded bytes before comparisons
            for c in re.findall(r"%[a-fA-F0-9]{2}", x):
                x = x.replace(c, c.upper())
            return x

        r = requests.Request("GET", url=url)
        p = r.prepare()
        assert normalize_percent_encode(p.url) == expected

    @pytest.mark.parametrize(
        "url",
        (
            b"http://*.google.com",
            b"http://*",
            "http://*.google.com",
            "http://*",
            "http://☃.net/",
        ),
    )
    def test_preparing_bad_url(self, url):
        r = requests.Request("GET", url=url)
        with pytest.raises(requests.exceptions.InvalidURL):
            r.prepare()

    @pytest.mark.parametrize("url, exception", (("http://:1", InvalidURL),))
    def test_redirecting_to_bad_url(self, httpbin, url, exception):
        with pytest.raises(exception):
            requests.get(httpbin("redirect-to"), params={"url": url})

    @pytest.mark.parametrize(
        "input, expected",
        (
            (
                b"http+unix://%2Fvar%2Frun%2Fsocket/path%7E",
                "http+unix://%2Fvar%2Frun%2Fsocket/path~",
            ),
            (
                "http+unix://%2Fvar%2Frun%2Fsocket/path%7E",
                "http+unix://%2Fvar%2Frun%2Fsocket/path~",
            ),
            (
                b"mailto:user@example.org",
                "mailto:user@example.org",
            ),
            (
                "mailto:user@example.org",
                "mailto:user@example.org",
            ),
            (
                b"data:SSDimaUgUHl0aG9uIQ==",
                "data:SSDimaUgUHl0aG9uIQ==",
            ),
        ),
    )
    def test_url_mutation(self, input, expected):
        """
        This test validates that we correctly exclude some URLs from
        preparation, and that we handle others. Specifically, it tests that
        any URL whose scheme doesn't begin with "http" is left alone, and
        those whose scheme *does* begin with "http" are mutated.
        """
        r = requests.Request("GET", url=input)
        p = r.prepare()
        assert p.url == expected

    @pytest.mark.parametrize(
        "input, params, expected",
        (
            (
                b"http+unix://%2Fvar%2Frun%2Fsocket/path",
                {"key": "value"},
                "http+unix://%2Fvar%2Frun%2Fsocket/path?key=value",
            ),
            (
                "http+unix://%2Fvar%2Frun%2Fsocket/path",
                {"key": "value"},
                "http+unix://%2Fvar%2Frun%2Fsocket/path?key=value",
            ),
            (
                b"mailto:user@example.org",
                {"key": "value"},
                "mailto:user@example.org",
            ),
            (
                "mailto:user@example.org",
                {"key": "value"},
                "mailto:user@example.org",
            ),
        ),
    )
    def test_parameters_for_nonstandard_schemes(self, input, params, expected):
        """
        Setting parameters for nonstandard schemes is allowed if those schemes
        begin with "http", and is forbidden otherwise.
        """
        r = requests.Request("GET", url=input, params=params)
        p = r.prepare()
        assert p.url == expected

    def test_post_json_nan(self, httpbin):
        data = {"foo": float("nan")}
        with pytest.raises(requests.exceptions.InvalidJSONError):
            requests.post(httpbin("post"), json=data)

    def test_json_decode_compatibility(self, httpbin):
        r = requests.get(httpbin("bytes/20"))
        with pytest.raises(requests.exceptions.JSONDecodeError) as excinfo:
            r.json()
        assert isinstance(excinfo.value, RequestException)
        assert isinstance(excinfo.value, JSONDecodeError)
        assert r.text not in str(excinfo.value)

    def test_json_decode_persists_doc_attr(self, httpbin):
        r = requests.get(httpbin("bytes/20"))
        with pytest.raises(requests.exceptions.JSONDecodeError) as excinfo:
            r.json()
        assert excinfo.value.doc == r.text

    def test_status_code_425(self):
        r1 = requests.codes.get("TOO_EARLY")
        r2 = requests.codes.get("too_early")
        r3 = requests.codes.get("UNORDERED")
        r4 = requests.codes.get("unordered")
        r5 = requests.codes.get("UNORDERED_COLLECTION")
        r6 = requests.codes.get("unordered_collection")

        assert r1 == 425
        assert r2 == 425
        assert r3 == 425
        assert r4 == 425
        assert r5 == 425
        assert r6 == 425

    def test_different_connection_pool_for_tls_settings_verify_True(self):
        def response_handler(sock):
            consume_socket_content(sock, timeout=0.5)
            sock.send(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 18\r\n\r\n"
                b'\xff\xfe{\x00"\x00K0"\x00=\x00"\x00\xab0"\x00\r\n'
            )

        s = requests.Session()
        close_server = threading.Event()
        server = TLSServer(
            handler=response_handler,
            wait_to_close_event=close_server,
            requests_to_handle=3,
            cert_chain="tests/certs/expired/server/server.pem",
            keyfile="tests/certs/expired/server/server.key",
        )

        with server as (host, port):
            url = f"https://{host}:{port}"
            r1 = s.get(url, verify=False)
            assert r1.status_code == 200

            # Cannot verify self-signed certificate
            with pytest.raises(requests.exceptions.SSLError):
                s.get(url)

            close_server.set()
        assert 2 == len(s.adapters["https://"].poolmanager.pools)

    def test_different_connection_pool_for_tls_settings_verify_bundle_expired_cert(
        self,
    ):
        def response_handler(sock):
            consume_socket_content(sock, timeout=0.5)
            sock.send(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 18\r\n\r\n"
                b'\xff\xfe{\x00"\x00K0"\x00=\x00"\x00\xab0"\x00\r\n'
            )

        s = requests.Session()
        close_server = threading.Event()
        server = TLSServer(
            handler=response_handler,
            wait_to_close_event=close_server,
            requests_to_handle=3,
            cert_chain="tests/certs/expired/server/server.pem",
            keyfile="tests/certs/expired/server/server.key",
        )

        with server as (host, port):
            url = f"https://{host}:{port}"
            r1 = s.get(url, verify=False)
            assert r1.status_code == 200

            # Has right trust bundle, but certificate expired
            with pytest.raises(requests.exceptions.SSLError):
                s.get(url, verify="tests/certs/expired/ca/ca.crt")

            close_server.set()
        assert 2 == len(s.adapters["https://"].poolmanager.pools)

    def test_different_connection_pool_for_tls_settings_verify_bundle_unexpired_cert(
        self,
    ):
        def response_handler(sock):
            consume_socket_content(sock, timeout=0.5)
            sock.send(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 18\r\n\r\n"
                b'\xff\xfe{\x00"\x00K0"\x00=\x00"\x00\xab0"\x00\r\n'
            )

        s = requests.Session()
        close_server = threading.Event()
        server = TLSServer(
            handler=response_handler,
            wait_to_close_event=close_server,
            requests_to_handle=3,
            cert_chain="tests/certs/valid/server/server.pem",
            keyfile="tests/certs/valid/server/server.key",
        )

        with server as (host, port):
            url = f"https://{host}:{port}"
            r1 = s.get(url, verify=False)
            assert r1.status_code == 200

            r2 = s.get(url, verify="tests/certs/valid/ca/ca.crt")
            assert r2.status_code == 200

            close_server.set()
        assert 2 == len(s.adapters["https://"].poolmanager.pools)

    def test_different_connection_pool_for_mtls_settings(self):
        client_cert = None

        def response_handler(sock):
            nonlocal client_cert
            client_cert = sock.getpeercert()
            consume_socket_content(sock, timeout=0.5)
            sock.send(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 18\r\n\r\n"
                b'\xff\xfe{\x00"\x00K0"\x00=\x00"\x00\xab0"\x00\r\n'
            )

        s = requests.Session()
        close_server = threading.Event()
        server = TLSServer(
            handler=response_handler,
            wait_to_close_event=close_server,
            requests_to_handle=2,
            cert_chain="tests/certs/expired/server/server.pem",
            keyfile="tests/certs/expired/server/server.key",
            mutual_tls=True,
            cacert="tests/certs/expired/ca/ca.crt",
        )

        cert = (
            "tests/certs/mtls/client/client.pem",
            "tests/certs/mtls/client/client.key",
        )
        with server as (host, port):
            url = f"https://{host}:{port}"
            r1 = s.get(url, verify=False, cert=cert)
            assert r1.status_code == 200
            with pytest.raises(requests.exceptions.SSLError):
                s.get(url, cert=cert)
            close_server.set()

        assert client_cert is not None


def test_content_length_for_bytes_data(httpbin):
    data = "This is a string containing multi-byte UTF-8 ☃️"
    encoded_data = data.encode("utf-8")
    length = str(len(encoded_data))
    req = requests.Request("POST", httpbin("post"), data=encoded_data)
    p = req.prepare()

    assert p.headers["Content-Length"] == length


@pytest.mark.skipif(
    is_urllib3_1,
    reason="urllib3 2.x encodes all strings to utf-8, urllib3 1.x uses latin-1",
)
def test_content_length_for_string_data_counts_bytes(httpbin):
    data = "This is a string containing multi-byte UTF-8 ☃️"
    length = str(len(data.encode("utf-8")))
    req = requests.Request("POST", httpbin("post"), data=data)
    p = req.prepare()

    assert p.headers["Content-Length"] == length


def test_json_decode_errors_are_serializable_deserializable():
    json_decode_error = requests.exceptions.JSONDecodeError(
        "Extra data",
        '{"responseCode":["706"],"data":null}{"responseCode":["706"],"data":null}',
        36,
    )
    deserialized_error = pickle.loads(pickle.dumps(json_decode_error))
    assert repr(json_decode_error) == repr(deserialized_error)


# === tests/utils.py ===
import contextlib
import os


@contextlib.contextmanager
def override_environ(**kwargs):
    save_env = dict(os.environ)
    for key, value in kwargs.items():
        if value is None:
            del os.environ[key]
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(save_env)


# === tests/test_adapters.py ===
import requests.adapters


def test_request_url_trims_leading_path_separators():
    """See also https://github.com/psf/requests/issues/6643."""
    a = requests.adapters.HTTPAdapter()
    p = requests.Request(method="GET", url="http://127.0.0.1:10000//v:h").prepare()
    assert "/v:h" == a.request_url(p, {})


# === tests/test_help.py ===
from unittest import mock

from requests.help import info


def test_system_ssl():
    """Verify we're actually setting system_ssl when it should be available."""
    assert info()["system_ssl"]["version"] != ""


class VersionedPackage:
    def __init__(self, version):
        self.__version__ = version


def test_idna_without_version_attribute():
    """Older versions of IDNA don't provide a __version__ attribute, verify
    that if we have such a package, we don't blow up.
    """
    with mock.patch("requests.help.idna", new=None):
        assert info()["idna"] == {"version": ""}


def test_idna_with_version_attribute():
    """Verify we're actually setting idna version when it should be available."""
    with mock.patch("requests.help.idna", new=VersionedPackage("2.6")):
        assert info()["idna"] == {"version": "2.6"}


# === tests/test_lowlevel.py ===
import threading

import pytest
from tests.testserver.server import Server, consume_socket_content

import requests
from requests.compat import JSONDecodeError

from .utils import override_environ


def echo_response_handler(sock):
    """Simple handler that will take request and echo it back to requester."""
    request_content = consume_socket_content(sock, timeout=0.5)

    text_200 = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: %d\r\n\r\n"
        b"%s"
    ) % (len(request_content), request_content)
    sock.send(text_200)


def test_chunked_upload():
    """can safely send generators"""
    close_server = threading.Event()
    server = Server.basic_response_server(wait_to_close_event=close_server)
    data = iter([b"a", b"b", b"c"])

    with server as (host, port):
        url = f"http://{host}:{port}/"
        r = requests.post(url, data=data, stream=True)
        close_server.set()  # release server block

    assert r.status_code == 200
    assert r.request.headers["Transfer-Encoding"] == "chunked"


def test_chunked_encoding_error():
    """get a ChunkedEncodingError if the server returns a bad response"""

    def incomplete_chunked_response_handler(sock):
        request_content = consume_socket_content(sock, timeout=0.5)

        # The server never ends the request and doesn't provide any valid chunks
        sock.send(
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
        )

        return request_content

    close_server = threading.Event()
    server = Server(incomplete_chunked_response_handler)

    with server as (host, port):
        url = f"http://{host}:{port}/"
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            requests.get(url)
        close_server.set()  # release server block


def test_chunked_upload_uses_only_specified_host_header():
    """Ensure we use only the specified Host header for chunked requests."""
    close_server = threading.Event()
    server = Server(echo_response_handler, wait_to_close_event=close_server)

    data = iter([b"a", b"b", b"c"])
    custom_host = "sample-host"

    with server as (host, port):
        url = f"http://{host}:{port}/"
        r = requests.post(url, data=data, headers={"Host": custom_host}, stream=True)
        close_server.set()  # release server block

    expected_header = b"Host: %s\r\n" % custom_host.encode("utf-8")
    assert expected_header in r.content
    assert r.content.count(b"Host: ") == 1


def test_chunked_upload_doesnt_skip_host_header():
    """Ensure we don't omit all Host headers with chunked requests."""
    close_server = threading.Event()
    server = Server(echo_response_handler, wait_to_close_event=close_server)

    data = iter([b"a", b"b", b"c"])

    with server as (host, port):
        expected_host = f"{host}:{port}"
        url = f"http://{host}:{port}/"
        r = requests.post(url, data=data, stream=True)
        close_server.set()  # release server block

    expected_header = b"Host: %s\r\n" % expected_host.encode("utf-8")
    assert expected_header in r.content
    assert r.content.count(b"Host: ") == 1


def test_conflicting_content_lengths():
    """Ensure we correctly throw an InvalidHeader error if multiple
    conflicting Content-Length headers are returned.
    """

    def multiple_content_length_response_handler(sock):
        request_content = consume_socket_content(sock, timeout=0.5)
        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: 16\r\n"
            b"Content-Length: 32\r\n\r\n"
            b"-- Bad Actor -- Original Content\r\n"
        )
        sock.send(response)

        return request_content

    close_server = threading.Event()
    server = Server(multiple_content_length_response_handler)

    with server as (host, port):
        url = f"http://{host}:{port}/"
        with pytest.raises(requests.exceptions.InvalidHeader):
            requests.get(url)
        close_server.set()


def test_digestauth_401_count_reset_on_redirect():
    """Ensure we correctly reset num_401_calls after a successful digest auth,
    followed by a 302 redirect to another digest auth prompt.

    See https://github.com/psf/requests/issues/1979.
    """
    text_401 = (b'HTTP/1.1 401 UNAUTHORIZED\r\n'
                b'Content-Length: 0\r\n'
                b'WWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d"'
                b', opaque="372825293d1c26955496c80ed6426e9e", '
                b'realm="me@kennethreitz.com", qop=auth\r\n\r\n')

    text_302 = (b'HTTP/1.1 302 FOUND\r\n'
                b'Content-Length: 0\r\n'
                b'Location: /\r\n\r\n')

    text_200 = (b'HTTP/1.1 200 OK\r\n'
                b'Content-Length: 0\r\n\r\n')

    expected_digest = (b'Authorization: Digest username="user", '
                       b'realm="me@kennethreitz.com", '
                       b'nonce="6bf5d6e4da1ce66918800195d6b9130d", uri="/"')

    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_response_handler(sock):
        # Respond to initial GET with a challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b"GET / HTTP/1.1")
        sock.send(text_401)

        # Verify we receive an Authorization header in response, then redirect.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert expected_digest in request_content
        sock.send(text_302)

        # Verify Authorization isn't sent to the redirected host,
        # then send another challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert b'Authorization:' not in request_content
        sock.send(text_401)

        # Verify Authorization is sent correctly again, and return 200 OK.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert expected_digest in request_content
        sock.send(text_200)

        return request_content

    close_server = threading.Event()
    server = Server(digest_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/'
        r = requests.get(url, auth=auth)
        # Verify server succeeded in authenticating.
        assert r.status_code == 200
        # Verify Authorization was sent in final request.
        assert 'Authorization' in r.request.headers
        assert r.request.headers['Authorization'].startswith('Digest ')
        # Verify redirect happened as we expected.
        assert r.history[0].status_code == 302
        close_server.set()


def test_digestauth_401_only_sent_once():
    """Ensure we correctly respond to a 401 challenge once, and then
    stop responding if challenged again.
    """
    text_401 = (b'HTTP/1.1 401 UNAUTHORIZED\r\n'
                b'Content-Length: 0\r\n'
                b'WWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d"'
                b', opaque="372825293d1c26955496c80ed6426e9e", '
                b'realm="me@kennethreitz.com", qop=auth\r\n\r\n')

    expected_digest = (b'Authorization: Digest username="user", '
                       b'realm="me@kennethreitz.com", '
                       b'nonce="6bf5d6e4da1ce66918800195d6b9130d", uri="/"')

    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_failed_response_handler(sock):
        # Respond to initial GET with a challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b"GET / HTTP/1.1")
        sock.send(text_401)

        # Verify we receive an Authorization header in response, then
        # challenge again.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert expected_digest in request_content
        sock.send(text_401)

        # Verify the client didn't respond to second challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content == b''

        return request_content

    close_server = threading.Event()
    server = Server(digest_failed_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/'
        r = requests.get(url, auth=auth)
        # Verify server didn't authenticate us.
        assert r.status_code == 401
        assert r.history[0].status_code == 401
        close_server.set()


def test_digestauth_only_on_4xx():
    """Ensure we only send digestauth on 4xx challenges.

    See https://github.com/psf/requests/issues/3772.
    """
    text_200_chal = (b'HTTP/1.1 200 OK\r\n'
                     b'Content-Length: 0\r\n'
                     b'WWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d"'
                     b', opaque="372825293d1c26955496c80ed6426e9e", '
                     b'realm="me@kennethreitz.com", qop=auth\r\n\r\n')

    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_response_handler(sock):
        # Respond to GET with a 200 containing www-authenticate header.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b"GET / HTTP/1.1")
        sock.send(text_200_chal)

        # Verify the client didn't respond with auth.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content == b''

        return request_content

    close_server = threading.Event()
    server = Server(digest_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/'
        r = requests.get(url, auth=auth)
        # Verify server didn't receive auth from us.
        assert r.status_code == 200
        assert len(r.history) == 0
        close_server.set()


_schemes_by_var_prefix = [
    ('http', ['http']),
    ('https', ['https']),
    ('all', ['http', 'https']),
]

_proxy_combos = []
for prefix, schemes in _schemes_by_var_prefix:
    for scheme in schemes:
        _proxy_combos.append((f"{prefix}_proxy", scheme))

_proxy_combos += [(var.upper(), scheme) for var, scheme in _proxy_combos]


@pytest.mark.parametrize("var,scheme", _proxy_combos)
def test_use_proxy_from_environment(httpbin, var, scheme):
    url = f"{scheme}://httpbin.org"
    fake_proxy = Server()  # do nothing with the requests; just close the socket
    with fake_proxy as (host, port):
        proxy_url = f"socks5://{host}:{port}"
        kwargs = {var: proxy_url}
        with override_environ(**kwargs):
            # fake proxy's lack of response will cause a ConnectionError
            with pytest.raises(requests.exceptions.ConnectionError):
                requests.get(url)

        # the fake proxy received a request
        assert len(fake_proxy.handler_results) == 1

        # it had actual content (not checking for SOCKS protocol for now)
        assert len(fake_proxy.handler_results[0]) > 0


def test_redirect_rfc1808_to_non_ascii_location():
    path = 'š'
    expected_path = b'%C5%A1'
    redirect_request = []  # stores the second request to the server

    def redirect_resp_handler(sock):
        consume_socket_content(sock, timeout=0.5)
        location = f'//{host}:{port}/{path}'
        sock.send(
            (
                b'HTTP/1.1 301 Moved Permanently\r\n'
                b'Content-Length: 0\r\n'
                b'Location: %s\r\n'
                b'\r\n'
            ) % location.encode('utf8')
        )
        redirect_request.append(consume_socket_content(sock, timeout=0.5))
        sock.send(b'HTTP/1.1 200 OK\r\n\r\n')

    close_server = threading.Event()
    server = Server(redirect_resp_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}'
        r = requests.get(url=url, allow_redirects=True)
        assert r.status_code == 200
        assert len(r.history) == 1
        assert r.history[0].status_code == 301
        assert redirect_request[0].startswith(b'GET /' + expected_path + b' HTTP/1.1')
        assert r.url == '{}/{}'.format(url, expected_path.decode('ascii'))

        close_server.set()


def test_fragment_not_sent_with_request():
    """Verify that the fragment portion of a URI isn't sent to the server."""
    close_server = threading.Event()
    server = Server(echo_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/path/to/thing/#view=edit&token=hunter2'
        r = requests.get(url)
        raw_request = r.content

        assert r.status_code == 200
        headers, body = raw_request.split(b'\r\n\r\n', 1)
        status_line, headers = headers.split(b'\r\n', 1)

        assert status_line == b'GET /path/to/thing/ HTTP/1.1'
        for frag in (b'view', b'edit', b'token', b'hunter2'):
            assert frag not in headers
            assert frag not in body

        close_server.set()


def test_fragment_update_on_redirect():
    """Verify we only append previous fragment if one doesn't exist on new
    location. If a new fragment is encountered in a Location header, it should
    be added to all subsequent requests.
    """

    def response_handler(sock):
        consume_socket_content(sock, timeout=0.5)
        sock.send(
            b'HTTP/1.1 302 FOUND\r\n'
            b'Content-Length: 0\r\n'
            b'Location: /get#relevant-section\r\n\r\n'
        )
        consume_socket_content(sock, timeout=0.5)
        sock.send(
            b'HTTP/1.1 302 FOUND\r\n'
            b'Content-Length: 0\r\n'
            b'Location: /final-url/\r\n\r\n'
        )
        consume_socket_content(sock, timeout=0.5)
        sock.send(
            b'HTTP/1.1 200 OK\r\n\r\n'
        )

    close_server = threading.Event()
    server = Server(response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/path/to/thing/#view=edit&token=hunter2'
        r = requests.get(url)

        assert r.status_code == 200
        assert len(r.history) == 2
        assert r.history[0].request.url == url

        # Verify we haven't overwritten the location with our previous fragment.
        assert r.history[1].request.url == f'http://{host}:{port}/get#relevant-section'
        # Verify previous fragment is used and not the original.
        assert r.url == f'http://{host}:{port}/final-url/#relevant-section'

        close_server.set()


def test_json_decode_compatibility_for_alt_utf_encodings():

    def response_handler(sock):
        consume_socket_content(sock, timeout=0.5)
        sock.send(
            b'HTTP/1.1 200 OK\r\n'
            b'Content-Length: 18\r\n\r\n'
            b'\xff\xfe{\x00"\x00K0"\x00=\x00"\x00\xab0"\x00\r\n'
        )

    close_server = threading.Event()
    server = Server(response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = f'http://{host}:{port}/'
        r = requests.get(url)
    r.encoding = None
    with pytest.raises(requests.exceptions.JSONDecodeError) as excinfo:
        r.json()
    assert isinstance(excinfo.value, requests.exceptions.RequestException)
    assert isinstance(excinfo.value, JSONDecodeError)
    assert r.text not in str(excinfo.value)


# === tests/testserver/server.py ===
import select
import socket
import ssl
import threading


def consume_socket_content(sock, timeout=0.5):
    chunks = 65536
    content = b""

    while True:
        more_to_read = select.select([sock], [], [], timeout)[0]
        if not more_to_read:
            break

        new_content = sock.recv(chunks)
        if not new_content:
            break

        content += new_content

    return content


class Server(threading.Thread):
    """Dummy server using for unit testing"""

    WAIT_EVENT_TIMEOUT = 5

    def __init__(
        self,
        handler=None,
        host="localhost",
        port=0,
        requests_to_handle=1,
        wait_to_close_event=None,
    ):
        super().__init__()

        self.handler = handler or consume_socket_content
        self.handler_results = []

        self.host = host
        self.port = port
        self.requests_to_handle = requests_to_handle

        self.wait_to_close_event = wait_to_close_event
        self.ready_event = threading.Event()
        self.stop_event = threading.Event()

    @classmethod
    def text_response_server(cls, text, request_timeout=0.5, **kwargs):
        def text_response_handler(sock):
            request_content = consume_socket_content(sock, timeout=request_timeout)
            sock.send(text.encode("utf-8"))

            return request_content

        return Server(text_response_handler, **kwargs)

    @classmethod
    def basic_response_server(cls, **kwargs):
        return cls.text_response_server(
            "HTTP/1.1 200 OK\r\n" + "Content-Length: 0\r\n\r\n", **kwargs
        )

    def run(self):
        try:
            self.server_sock = self._create_socket_and_bind()
            # in case self.port = 0
            self.port = self.server_sock.getsockname()[1]
            self.ready_event.set()
            self._handle_requests()

            if self.wait_to_close_event:
                self.wait_to_close_event.wait(self.WAIT_EVENT_TIMEOUT)
        finally:
            self.ready_event.set()  # just in case of exception
            self._close_server_sock_ignore_errors()
            self.stop_event.set()

    def _create_socket_and_bind(self):
        sock = socket.socket()
        sock.bind((self.host, self.port))
        sock.listen()
        return sock

    def _close_server_sock_ignore_errors(self):
        try:
            self.server_sock.close()
        except OSError:
            pass

    def _handle_requests(self):
        for _ in range(self.requests_to_handle):
            sock = self._accept_connection()
            if not sock:
                break

            handler_result = self.handler(sock)

            self.handler_results.append(handler_result)
            sock.close()

    def _accept_connection(self):
        try:
            ready, _, _ = select.select(
                [self.server_sock], [], [], self.WAIT_EVENT_TIMEOUT
            )
            if not ready:
                return None

            return self.server_sock.accept()[0]
        except OSError:
            return None

    def __enter__(self):
        self.start()
        if not self.ready_event.wait(self.WAIT_EVENT_TIMEOUT):
            raise RuntimeError("Timeout waiting for server to be ready.")
        return self.host, self.port

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.stop_event.wait(self.WAIT_EVENT_TIMEOUT)
        else:
            if self.wait_to_close_event:
                # avoid server from waiting for event timeouts
                # if an exception is found in the main thread
                self.wait_to_close_event.set()

        # ensure server thread doesn't get stuck waiting for connections
        self._close_server_sock_ignore_errors()
        self.join()
        return False  # allow exceptions to propagate


class TLSServer(Server):
    def __init__(
        self,
        *,
        handler=None,
        host="localhost",
        port=0,
        requests_to_handle=1,
        wait_to_close_event=None,
        cert_chain=None,
        keyfile=None,
        mutual_tls=False,
        cacert=None,
    ):
        super().__init__(
            handler=handler,
            host=host,
            port=port,
            requests_to_handle=requests_to_handle,
            wait_to_close_event=wait_to_close_event,
        )
        self.cert_chain = cert_chain
        self.keyfile = keyfile
        self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.ssl_context.load_cert_chain(self.cert_chain, keyfile=self.keyfile)
        self.mutual_tls = mutual_tls
        self.cacert = cacert
        if mutual_tls:
            # For simplicity, we're going to assume that the client cert is
            # issued by the same CA as our Server certificate
            self.ssl_context.verify_mode = ssl.CERT_OPTIONAL
            self.ssl_context.load_verify_locations(self.cacert)

    def _create_socket_and_bind(self):
        sock = socket.socket()
        sock = self.ssl_context.wrap_socket(sock, server_side=True)
        sock.bind((self.host, self.port))
        sock.listen()
        return sock


# === tests/testserver/__init__.py ===


# === docs/conf.py ===
# -*- coding: utf-8 -*-
#
# Requests documentation build configuration file, created by
# sphinx-quickstart on Fri Feb 19 00:05:47 2016.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# Insert Requests' path into the system.
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_themes"))

import requests


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = u"Requests"
copyright = u'MMXVIX. A <a href="https://kenreitz.org/projects">Kenneth Reitz</a> Project'
author = u"Kenneth Reitz"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = requests.__version__
# The full version, including alpha/beta/rc tags.
release = requests.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "flask_theme_support.FlaskyStyle"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "alabaster"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "show_powered_by": False,
    "github_user": "requests",
    "github_repo": "requests",
    "github_banner": True,
    "show_related": False,
    "note_bg": "#FFF59C",
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = False

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": ["sidebarintro.html", "sourcelink.html", "searchbox.html", "hacks.html"],
    "**": [
        "sidebarlogo.html",
        "localtoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
        "hacks.html",
    ],
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'ru', 'sv', 'tr'
# html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
# html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
# html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = "Requestsdoc"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
    # Latex figure (float) alignment
    #'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "Requests.tex", u"Requests Documentation", u"Kenneth Reitz", "manual")
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "requests", u"Requests Documentation", [author], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "Requests",
        u"Requests Documentation",
        author,
        "Requests",
        "One line description of project.",
        "Miscellaneous",
    )
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False


# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The basename for the epub file. It defaults to the project name.
# epub_basename = project

# The HTML theme for the epub output. Since the default themes are not
# optimized for small screen space, using the same theme for HTML and epub
# output is usually not wise. This defaults to 'epub', a theme designed to save
# visual space.
# epub_theme = 'epub'

# The language of the text. It defaults to the language option
# or 'en' if the language is not set.
# epub_language = ''

# The scheme of the identifier. Typical schemes are ISBN or URL.
# epub_scheme = ''

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
# epub_identifier = ''

# A unique identification for the text.
# epub_uid = ''

# A tuple containing the cover image and cover page html template filenames.
# epub_cover = ()

# A sequence of (type, uri, title) tuples for the guide element of content.opf.
# epub_guide = ()

# HTML files that should be inserted before the pages created by sphinx.
# The format is a list of tuples containing the path and title.
# epub_pre_files = []

# HTML files that should be inserted after the pages created by sphinx.
# The format is a list of tuples containing the path and title.
# epub_post_files = []

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# The depth of the table of contents in toc.ncx.
# epub_tocdepth = 3

# Allow duplicate toc entries.
# epub_tocdup = True

# Choose between 'default' and 'includehidden'.
# epub_tocscope = 'default'

# Fix unsupported image types using the Pillow.
# epub_fix_images = False

# Scale large images.
# epub_max_image_width = 0

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# epub_show_urls = 'inline'

# If false, no index is generated.
# epub_use_index = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "urllib3": ("https://urllib3.readthedocs.io/en/latest", None),
}


# === docs/_themes/flask_theme_support.py ===
# flasky extensions.  flasky pygments style based on tango style
from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, \
     Number, Operator, Generic, Whitespace, Punctuation, Other, Literal


class FlaskyStyle(Style):
    background_color = "#f8f8f8"
    default_style = ""

    styles = {
        # No corresponding class for the following:
        #Text:                     "", # class:  ''
        Whitespace:                "underline #f8f8f8",      # class: 'w'
        Error:                     "#a40000 border:#ef2929", # class: 'err'
        Other:                     "#000000",                # class 'x'

        Comment:                   "italic #8f5902", # class: 'c'
        Comment.Preproc:           "noitalic",       # class: 'cp'

        Keyword:                   "bold #004461",   # class: 'k'
        Keyword.Constant:          "bold #004461",   # class: 'kc'
        Keyword.Declaration:       "bold #004461",   # class: 'kd'
        Keyword.Namespace:         "bold #004461",   # class: 'kn'
        Keyword.Pseudo:            "bold #004461",   # class: 'kp'
        Keyword.Reserved:          "bold #004461",   # class: 'kr'
        Keyword.Type:              "bold #004461",   # class: 'kt'

        Operator:                  "#582800",   # class: 'o'
        Operator.Word:             "bold #004461",   # class: 'ow' - like keywords

        Punctuation:               "bold #000000",   # class: 'p'

        # because special names such as Name.Class, Name.Function, etc.
        # are not recognized as such later in the parsing, we choose them
        # to look the same as ordinary variables.
        Name:                      "#000000",        # class: 'n'
        Name.Attribute:            "#c4a000",        # class: 'na' - to be revised
        Name.Builtin:              "#004461",        # class: 'nb'
        Name.Builtin.Pseudo:       "#3465a4",        # class: 'bp'
        Name.Class:                "#000000",        # class: 'nc' - to be revised
        Name.Constant:             "#000000",        # class: 'no' - to be revised
        Name.Decorator:            "#888",           # class: 'nd' - to be revised
        Name.Entity:               "#ce5c00",        # class: 'ni'
        Name.Exception:            "bold #cc0000",   # class: 'ne'
        Name.Function:             "#000000",        # class: 'nf'
        Name.Property:             "#000000",        # class: 'py'
        Name.Label:                "#f57900",        # class: 'nl'
        Name.Namespace:            "#000000",        # class: 'nn' - to be revised
        Name.Other:                "#000000",        # class: 'nx'
        Name.Tag:                  "bold #004461",   # class: 'nt' - like a keyword
        Name.Variable:             "#000000",        # class: 'nv' - to be revised
        Name.Variable.Class:       "#000000",        # class: 'vc' - to be revised
        Name.Variable.Global:      "#000000",        # class: 'vg' - to be revised
        Name.Variable.Instance:    "#000000",        # class: 'vi' - to be revised

        Number:                    "#990000",        # class: 'm'

        Literal:                   "#000000",        # class: 'l'
        Literal.Date:              "#000000",        # class: 'ld'

        String:                    "#4e9a06",        # class: 's'
        String.Backtick:           "#4e9a06",        # class: 'sb'
        String.Char:               "#4e9a06",        # class: 'sc'
        String.Doc:                "italic #8f5902", # class: 'sd' - like a comment
        String.Double:             "#4e9a06",        # class: 's2'
        String.Escape:             "#4e9a06",        # class: 'se'
        String.Heredoc:            "#4e9a06",        # class: 'sh'
        String.Interpol:           "#4e9a06",        # class: 'si'
        String.Other:              "#4e9a06",        # class: 'sx'
        String.Regex:              "#4e9a06",        # class: 'sr'
        String.Single:             "#4e9a06",        # class: 's1'
        String.Symbol:             "#4e9a06",        # class: 'ss'

        Generic:                   "#000000",        # class: 'g'
        Generic.Deleted:           "#a40000",        # class: 'gd'
        Generic.Emph:              "italic #000000", # class: 'ge'
        Generic.Error:             "#ef2929",        # class: 'gr'
        Generic.Heading:           "bold #000080",   # class: 'gh'
        Generic.Inserted:          "#00A000",        # class: 'gi'
        Generic.Output:            "#888",           # class: 'go'
        Generic.Prompt:            "#745334",        # class: 'gp'
        Generic.Strong:            "bold #000000",   # class: 'gs'
        Generic.Subheading:        "bold #800080",   # class: 'gu'
        Generic.Traceback:         "bold #a40000",   # class: 'gt'
    }


# === src/requests/cookies.py ===
"""
requests.cookies
~~~~~~~~~~~~~~~~

Compatibility code to be able to use `http.cookiejar.CookieJar` with requests.

requests.utils imports from here, so be careful with imports.
"""

import calendar
import copy
import time

from ._internal_utils import to_native_string
from .compat import Morsel, MutableMapping, cookielib, urlparse, urlunparse

try:
    import threading
except ImportError:
    import dummy_threading as threading


class MockRequest:
    """Wraps a `requests.Request` to mimic a `urllib2.Request`.

    The code in `http.cookiejar.CookieJar` expects this interface in order to correctly
    manage cookie policies, i.e., determine whether a cookie can be set, given the
    domains of the request and the cookie.

    The original request object is read-only. The client is responsible for collecting
    the new headers via `get_new_headers()` and interpreting them appropriately. You
    probably want `get_cookie_header`, defined below.
    """

    def __init__(self, request):
        self._r = request
        self._new_headers = {}
        self.type = urlparse(self._r.url).scheme

    def get_type(self):
        return self.type

    def get_host(self):
        return urlparse(self._r.url).netloc

    def get_origin_req_host(self):
        return self.get_host()

    def get_full_url(self):
        # Only return the response's URL if the user hadn't set the Host
        # header
        if not self._r.headers.get("Host"):
            return self._r.url
        # If they did set it, retrieve it and reconstruct the expected domain
        host = to_native_string(self._r.headers["Host"], encoding="utf-8")
        parsed = urlparse(self._r.url)
        # Reconstruct the URL as we expect it
        return urlunparse(
            [
                parsed.scheme,
                host,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            ]
        )

    def is_unverifiable(self):
        return True

    def has_header(self, name):
        return name in self._r.headers or name in self._new_headers

    def get_header(self, name, default=None):
        return self._r.headers.get(name, self._new_headers.get(name, default))

    def add_header(self, key, val):
        """cookiejar has no legitimate use for this method; add it back if you find one."""
        raise NotImplementedError(
            "Cookie headers should be added with add_unredirected_header()"
        )

    def add_unredirected_header(self, name, value):
        self._new_headers[name] = value

    def get_new_headers(self):
        return self._new_headers

    @property
    def unverifiable(self):
        return self.is_unverifiable()

    @property
    def origin_req_host(self):
        return self.get_origin_req_host()

    @property
    def host(self):
        return self.get_host()


class MockResponse:
    """Wraps a `httplib.HTTPMessage` to mimic a `urllib.addinfourl`.

    ...what? Basically, expose the parsed HTTP headers from the server response
    the way `http.cookiejar` expects to see them.
    """

    def __init__(self, headers):
        """Make a MockResponse for `cookiejar` to read.

        :param headers: a httplib.HTTPMessage or analogous carrying the headers
        """
        self._headers = headers

    def info(self):
        return self._headers

    def getheaders(self, name):
        self._headers.getheaders(name)


def extract_cookies_to_jar(jar, request, response):
    """Extract the cookies from the response into a CookieJar.

    :param jar: http.cookiejar.CookieJar (not necessarily a RequestsCookieJar)
    :param request: our own requests.Request object
    :param response: urllib3.HTTPResponse object
    """
    if not (hasattr(response, "_original_response") and response._original_response):
        return
    # the _original_response field is the wrapped httplib.HTTPResponse object,
    req = MockRequest(request)
    # pull out the HTTPMessage with the headers and put it in the mock:
    res = MockResponse(response._original_response.msg)
    jar.extract_cookies(res, req)


def get_cookie_header(jar, request):
    """
    Produce an appropriate Cookie header string to be sent with `request`, or None.

    :rtype: str
    """
    r = MockRequest(request)
    jar.add_cookie_header(r)
    return r.get_new_headers().get("Cookie")


def remove_cookie_by_name(cookiejar, name, domain=None, path=None):
    """Unsets a cookie by name, by default over all domains and paths.

    Wraps CookieJar.clear(), is O(n).
    """
    clearables = []
    for cookie in cookiejar:
        if cookie.name != name:
            continue
        if domain is not None and domain != cookie.domain:
            continue
        if path is not None and path != cookie.path:
            continue
        clearables.append((cookie.domain, cookie.path, cookie.name))

    for domain, path, name in clearables:
        cookiejar.clear(domain, path, name)


class CookieConflictError(RuntimeError):
    """There are two cookies that meet the criteria specified in the cookie jar.
    Use .get and .set and include domain and path args in order to be more specific.
    """


class RequestsCookieJar(cookielib.CookieJar, MutableMapping):
    """Compatibility class; is a http.cookiejar.CookieJar, but exposes a dict
    interface.

    This is the CookieJar we create by default for requests and sessions that
    don't specify one, since some clients may expect response.cookies and
    session.cookies to support dict operations.

    Requests does not use the dict interface internally; it's just for
    compatibility with external client code. All requests code should work
    out of the box with externally provided instances of ``CookieJar``, e.g.
    ``LWPCookieJar`` and ``FileCookieJar``.

    Unlike a regular CookieJar, this class is pickleable.

    .. warning:: dictionary operations that are normally O(1) may be O(n).
    """

    def get(self, name, default=None, domain=None, path=None):
        """Dict-like get() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.

        .. warning:: operation is O(n), not O(1).
        """
        try:
            return self._find_no_duplicates(name, domain, path)
        except KeyError:
            return default

    def set(self, name, value, **kwargs):
        """Dict-like set() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.
        """
        # support client code that unsets cookies by assignment of a None value:
        if value is None:
            remove_cookie_by_name(
                self, name, domain=kwargs.get("domain"), path=kwargs.get("path")
            )
            return

        if isinstance(value, Morsel):
            c = morsel_to_cookie(value)
        else:
            c = create_cookie(name, value, **kwargs)
        self.set_cookie(c)
        return c

    def iterkeys(self):
        """Dict-like iterkeys() that returns an iterator of names of cookies
        from the jar.

        .. seealso:: itervalues() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.name

    def keys(self):
        """Dict-like keys() that returns a list of names of cookies from the
        jar.

        .. seealso:: values() and items().
        """
        return list(self.iterkeys())

    def itervalues(self):
        """Dict-like itervalues() that returns an iterator of values of cookies
        from the jar.

        .. seealso:: iterkeys() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.value

    def values(self):
        """Dict-like values() that returns a list of values of cookies from the
        jar.

        .. seealso:: keys() and items().
        """
        return list(self.itervalues())

    def iteritems(self):
        """Dict-like iteritems() that returns an iterator of name-value tuples
        from the jar.

        .. seealso:: iterkeys() and itervalues().
        """
        for cookie in iter(self):
            yield cookie.name, cookie.value

    def items(self):
        """Dict-like items() that returns a list of name-value tuples from the
        jar. Allows client-code to call ``dict(RequestsCookieJar)`` and get a
        vanilla python dict of key value pairs.

        .. seealso:: keys() and values().
        """
        return list(self.iteritems())

    def list_domains(self):
        """Utility method to list all the domains in the jar."""
        domains = []
        for cookie in iter(self):
            if cookie.domain not in domains:
                domains.append(cookie.domain)
        return domains

    def list_paths(self):
        """Utility method to list all the paths in the jar."""
        paths = []
        for cookie in iter(self):
            if cookie.path not in paths:
                paths.append(cookie.path)
        return paths

    def multiple_domains(self):
        """Returns True if there are multiple domains in the jar.
        Returns False otherwise.

        :rtype: bool
        """
        domains = []
        for cookie in iter(self):
            if cookie.domain is not None and cookie.domain in domains:
                return True
            domains.append(cookie.domain)
        return False  # there is only one domain in jar

    def get_dict(self, domain=None, path=None):
        """Takes as an argument an optional domain and path and returns a plain
        old Python dict of name-value pairs of cookies that meet the
        requirements.

        :rtype: dict
        """
        dictionary = {}
        for cookie in iter(self):
            if (domain is None or cookie.domain == domain) and (
                path is None or cookie.path == path
            ):
                dictionary[cookie.name] = cookie.value
        return dictionary

    def __contains__(self, name):
        try:
            return super().__contains__(name)
        except CookieConflictError:
            return True

    def __getitem__(self, name):
        """Dict-like __getitem__() for compatibility with client code. Throws
        exception if there are more than one cookie with name. In that case,
        use the more explicit get() method instead.

        .. warning:: operation is O(n), not O(1).
        """
        return self._find_no_duplicates(name)

    def __setitem__(self, name, value):
        """Dict-like __setitem__ for compatibility with client code. Throws
        exception if there is already a cookie of that name in the jar. In that
        case, use the more explicit set() method instead.
        """
        self.set(name, value)

    def __delitem__(self, name):
        """Deletes a cookie given a name. Wraps ``http.cookiejar.CookieJar``'s
        ``remove_cookie_by_name()``.
        """
        remove_cookie_by_name(self, name)

    def set_cookie(self, cookie, *args, **kwargs):
        if (
            hasattr(cookie.value, "startswith")
            and cookie.value.startswith('"')
            and cookie.value.endswith('"')
        ):
            cookie.value = cookie.value.replace('\\"', "")
        return super().set_cookie(cookie, *args, **kwargs)

    def update(self, other):
        """Updates this jar with cookies from another CookieJar or dict-like"""
        if isinstance(other, cookielib.CookieJar):
            for cookie in other:
                self.set_cookie(copy.copy(cookie))
        else:
            super().update(other)

    def _find(self, name, domain=None, path=None):
        """Requests uses this method internally to get cookie values.

        If there are conflicting cookies, _find arbitrarily chooses one.
        See _find_no_duplicates if you want an exception thrown if there are
        conflicting cookies.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :return: cookie.value
        """
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        return cookie.value

        raise KeyError(f"name={name!r}, domain={domain!r}, path={path!r}")

    def _find_no_duplicates(self, name, domain=None, path=None):
        """Both ``__get_item__`` and ``get`` call this function: it's never
        used elsewhere in Requests.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :raises KeyError: if cookie is not found
        :raises CookieConflictError: if there are multiple cookies
            that match name and optionally domain and path
        :return: cookie.value
        """
        toReturn = None
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        if toReturn is not None:
                            # if there are multiple cookies that meet passed in criteria
                            raise CookieConflictError(
                                f"There are multiple cookies with name, {name!r}"
                            )
                        # we will eventually return this as long as no cookie conflict
                        toReturn = cookie.value

        if toReturn:
            return toReturn
        raise KeyError(f"name={name!r}, domain={domain!r}, path={path!r}")

    def __getstate__(self):
        """Unlike a normal CookieJar, this class is pickleable."""
        state = self.__dict__.copy()
        # remove the unpickleable RLock object
        state.pop("_cookies_lock")
        return state

    def __setstate__(self, state):
        """Unlike a normal CookieJar, this class is pickleable."""
        self.__dict__.update(state)
        if "_cookies_lock" not in self.__dict__:
            self._cookies_lock = threading.RLock()

    def copy(self):
        """Return a copy of this RequestsCookieJar."""
        new_cj = RequestsCookieJar()
        new_cj.set_policy(self.get_policy())
        new_cj.update(self)
        return new_cj

    def get_policy(self):
        """Return the CookiePolicy instance used."""
        return self._policy


def _copy_cookie_jar(jar):
    if jar is None:
        return None

    if hasattr(jar, "copy"):
        # We're dealing with an instance of RequestsCookieJar
        return jar.copy()
    # We're dealing with a generic CookieJar instance
    new_jar = copy.copy(jar)
    new_jar.clear()
    for cookie in jar:
        new_jar.set_cookie(copy.copy(cookie))
    return new_jar


def create_cookie(name, value, **kwargs):
    """Make a cookie from underspecified parameters.

    By default, the pair of `name` and `value` will be set for the domain ''
    and sent on every request (this is sometimes called a "supercookie").
    """
    result = {
        "version": 0,
        "name": name,
        "value": value,
        "port": None,
        "domain": "",
        "path": "/",
        "secure": False,
        "expires": None,
        "discard": True,
        "comment": None,
        "comment_url": None,
        "rest": {"HttpOnly": None},
        "rfc2109": False,
    }

    badargs = set(kwargs) - set(result)
    if badargs:
        raise TypeError(
            f"create_cookie() got unexpected keyword arguments: {list(badargs)}"
        )

    result.update(kwargs)
    result["port_specified"] = bool(result["port"])
    result["domain_specified"] = bool(result["domain"])
    result["domain_initial_dot"] = result["domain"].startswith(".")
    result["path_specified"] = bool(result["path"])

    return cookielib.Cookie(**result)


def morsel_to_cookie(morsel):
    """Convert a Morsel object into a Cookie containing the one k/v pair."""

    expires = None
    if morsel["max-age"]:
        try:
            expires = int(time.time() + int(morsel["max-age"]))
        except ValueError:
            raise TypeError(f"max-age: {morsel['max-age']} must be integer")
    elif morsel["expires"]:
        time_template = "%a, %d-%b-%Y %H:%M:%S GMT"
        expires = calendar.timegm(time.strptime(morsel["expires"], time_template))
    return create_cookie(
        comment=morsel["comment"],
        comment_url=bool(morsel["comment"]),
        discard=False,
        domain=morsel["domain"],
        expires=expires,
        name=morsel.key,
        path=morsel["path"],
        port=None,
        rest={"HttpOnly": morsel["httponly"]},
        rfc2109=False,
        secure=bool(morsel["secure"]),
        value=morsel.value,
        version=morsel["version"] or 0,
    )


def cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True):
    """Returns a CookieJar from a key/value dictionary.

    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :param cookiejar: (optional) A cookiejar to add the cookies to.
    :param overwrite: (optional) If False, will not replace cookies
        already in the jar with new ones.
    :rtype: CookieJar
    """
    if cookiejar is None:
        cookiejar = RequestsCookieJar()

    if cookie_dict is not None:
        names_from_jar = [cookie.name for cookie in cookiejar]
        for name in cookie_dict:
            if overwrite or (name not in names_from_jar):
                cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))

    return cookiejar


def merge_cookies(cookiejar, cookies):
    """Add cookies to cookiejar and returns a merged CookieJar.

    :param cookiejar: CookieJar object to add the cookies to.
    :param cookies: Dictionary or CookieJar object to be added.
    :rtype: CookieJar
    """
    if not isinstance(cookiejar, cookielib.CookieJar):
        raise ValueError("You can only merge into CookieJar")

    if isinstance(cookies, dict):
        cookiejar = cookiejar_from_dict(cookies, cookiejar=cookiejar, overwrite=False)
    elif isinstance(cookies, cookielib.CookieJar):
        try:
            cookiejar.update(cookies)
        except AttributeError:
            for cookie_in_jar in cookies:
                cookiejar.set_cookie(cookie_in_jar)

    return cookiejar


# === src/requests/auth.py ===
"""
requests.auth
~~~~~~~~~~~~~

This module contains the authentication handlers for Requests.
"""

import hashlib
import os
import re
import threading
import time
import warnings
from base64 import b64encode

from ._internal_utils import to_native_string
from .compat import basestring, str, urlparse
from .cookies import extract_cookies_to_jar
from .utils import parse_dict_header

CONTENT_TYPE_FORM_URLENCODED = "application/x-www-form-urlencoded"
CONTENT_TYPE_MULTI_PART = "multipart/form-data"


def _basic_auth_str(username, password):
    """Returns a Basic Auth string."""

    # "I want us to put a big-ol' comment on top of it that
    # says that this behaviour is dumb but we need to preserve
    # it because people are relying on it."
    #    - Lukasa
    #
    # These are here solely to maintain backwards compatibility
    # for things like ints. This will be removed in 3.0.0.
    if not isinstance(username, basestring):
        warnings.warn(
            "Non-string usernames will no longer be supported in Requests "
            "3.0.0. Please convert the object you've passed in ({!r}) to "
            "a string or bytes object in the near future to avoid "
            "problems.".format(username),
            category=DeprecationWarning,
        )
        username = str(username)

    if not isinstance(password, basestring):
        warnings.warn(
            "Non-string passwords will no longer be supported in Requests "
            "3.0.0. Please convert the object you've passed in ({!r}) to "
            "a string or bytes object in the near future to avoid "
            "problems.".format(type(password)),
            category=DeprecationWarning,
        )
        password = str(password)
    # -- End Removal --

    if isinstance(username, str):
        username = username.encode("latin1")

    if isinstance(password, str):
        password = password.encode("latin1")

    authstr = "Basic " + to_native_string(
        b64encode(b":".join((username, password))).strip()
    )

    return authstr


class AuthBase:
    """Base class that all auth implementations derive from"""

    def __call__(self, r):
        raise NotImplementedError("Auth hooks must be callable.")


class HTTPBasicAuth(AuthBase):
    """Attaches HTTP Basic Authentication to the given Request object."""

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __eq__(self, other):
        return all(
            [
                self.username == getattr(other, "username", None),
                self.password == getattr(other, "password", None),
            ]
        )

    def __ne__(self, other):
        return not self == other

    def __call__(self, r):
        r.headers["Authorization"] = _basic_auth_str(self.username, self.password)
        return r


class HTTPProxyAuth(HTTPBasicAuth):
    """Attaches HTTP Proxy Authentication to a given Request object."""

    def __call__(self, r):
        r.headers["Proxy-Authorization"] = _basic_auth_str(self.username, self.password)
        return r


class HTTPDigestAuth(AuthBase):
    """Attaches HTTP Digest Authentication to the given Request object."""

    def __init__(self, username, password):
        self.username = username
        self.password = password
        # Keep state in per-thread local storage
        self._thread_local = threading.local()

    def init_per_thread_state(self):
        # Ensure state is initialized just once per-thread
        if not hasattr(self._thread_local, "init"):
            self._thread_local.init = True
            self._thread_local.last_nonce = ""
            self._thread_local.nonce_count = 0
            self._thread_local.chal = {}
            self._thread_local.pos = None
            self._thread_local.num_401_calls = None

    def build_digest_header(self, method, url):
        """
        :rtype: str
        """

        realm = self._thread_local.chal["realm"]
        nonce = self._thread_local.chal["nonce"]
        qop = self._thread_local.chal.get("qop")
        algorithm = self._thread_local.chal.get("algorithm")
        opaque = self._thread_local.chal.get("opaque")
        hash_utf8 = None

        if algorithm is None:
            _algorithm = "MD5"
        else:
            _algorithm = algorithm.upper()
        # lambdas assume digest modules are imported at the top level
        if _algorithm == "MD5" or _algorithm == "MD5-SESS":

            def md5_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.md5(x).hexdigest()

            hash_utf8 = md5_utf8
        elif _algorithm == "SHA":

            def sha_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha1(x).hexdigest()

            hash_utf8 = sha_utf8
        elif _algorithm == "SHA-256":

            def sha256_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha256(x).hexdigest()

            hash_utf8 = sha256_utf8
        elif _algorithm == "SHA-512":

            def sha512_utf8(x):
                if isinstance(x, str):
                    x = x.encode("utf-8")
                return hashlib.sha512(x).hexdigest()

            hash_utf8 = sha512_utf8

        KD = lambda s, d: hash_utf8(f"{s}:{d}")  # noqa:E731

        if hash_utf8 is None:
            return None

        # XXX not implemented yet
        entdig = None
        p_parsed = urlparse(url)
        #: path is request-uri defined in RFC 2616 which should not be empty
        path = p_parsed.path or "/"
        if p_parsed.query:
            path += f"?{p_parsed.query}"

        A1 = f"{self.username}:{realm}:{self.password}"
        A2 = f"{method}:{path}"

        HA1 = hash_utf8(A1)
        HA2 = hash_utf8(A2)

        if nonce == self._thread_local.last_nonce:
            self._thread_local.nonce_count += 1
        else:
            self._thread_local.nonce_count = 1
        ncvalue = f"{self._thread_local.nonce_count:08x}"
        s = str(self._thread_local.nonce_count).encode("utf-8")
        s += nonce.encode("utf-8")
        s += time.ctime().encode("utf-8")
        s += os.urandom(8)

        cnonce = hashlib.sha1(s).hexdigest()[:16]
        if _algorithm == "MD5-SESS":
            HA1 = hash_utf8(f"{HA1}:{nonce}:{cnonce}")

        if not qop:
            respdig = KD(HA1, f"{nonce}:{HA2}")
        elif qop == "auth" or "auth" in qop.split(","):
            noncebit = f"{nonce}:{ncvalue}:{cnonce}:auth:{HA2}"
            respdig = KD(HA1, noncebit)
        else:
            # XXX handle auth-int.
            return None

        self._thread_local.last_nonce = nonce

        # XXX should the partial digests be encoded too?
        base = (
            f'username="{self.username}", realm="{realm}", nonce="{nonce}", '
            f'uri="{path}", response="{respdig}"'
        )
        if opaque:
            base += f', opaque="{opaque}"'
        if algorithm:
            base += f', algorithm="{algorithm}"'
        if entdig:
            base += f', digest="{entdig}"'
        if qop:
            base += f', qop="auth", nc={ncvalue}, cnonce="{cnonce}"'

        return f"Digest {base}"

    def handle_redirect(self, r, **kwargs):
        """Reset num_401_calls counter on redirects."""
        if r.is_redirect:
            self._thread_local.num_401_calls = 1

    def handle_401(self, r, **kwargs):
        """
        Takes the given response and tries digest-auth, if needed.

        :rtype: requests.Response
        """

        # If response is not 4xx, do not auth
        # See https://github.com/psf/requests/issues/3772
        if not 400 <= r.status_code < 500:
            self._thread_local.num_401_calls = 1
            return r

        if self._thread_local.pos is not None:
            # Rewind the file position indicator of the body to where
            # it was to resend the request.
            r.request.body.seek(self._thread_local.pos)
        s_auth = r.headers.get("www-authenticate", "")

        if "digest" in s_auth.lower() and self._thread_local.num_401_calls < 2:
            self._thread_local.num_401_calls += 1
            pat = re.compile(r"digest ", flags=re.IGNORECASE)
            self._thread_local.chal = parse_dict_header(pat.sub("", s_auth, count=1))

            # Consume content and release the original connection
            # to allow our new request to reuse the same one.
            r.content
            r.close()
            prep = r.request.copy()
            extract_cookies_to_jar(prep._cookies, r.request, r.raw)
            prep.prepare_cookies(prep._cookies)

            prep.headers["Authorization"] = self.build_digest_header(
                prep.method, prep.url
            )
            _r = r.connection.send(prep, **kwargs)
            _r.history.append(r)
            _r.request = prep

            return _r

        self._thread_local.num_401_calls = 1
        return r

    def __call__(self, r):
        # Initialize per-thread state, if needed
        self.init_per_thread_state()
        # If we have a saved nonce, skip the 401
        if self._thread_local.last_nonce:
            r.headers["Authorization"] = self.build_digest_header(r.method, r.url)
        try:
            self._thread_local.pos = r.body.tell()
        except AttributeError:
            # In the case of HTTPDigestAuth being reused and the body of
            # the previous request was a file-like object, pos has the
            # file position of the previous body. Ensure it's set to
            # None.
            self._thread_local.pos = None
        r.register_hook("response", self.handle_401)
        r.register_hook("response", self.handle_redirect)
        self._thread_local.num_401_calls = 1

        return r

    def __eq__(self, other):
        return all(
            [
                self.username == getattr(other, "username", None),
                self.password == getattr(other, "password", None),
            ]
        )

    def __ne__(self, other):
        return not self == other


# === src/requests/sessions.py ===
"""
requests.sessions
~~~~~~~~~~~~~~~~~

This module provides a Session object to manage and persist settings across
requests (cookies, auth, proxies).
"""
import os
import sys
import time
from collections import OrderedDict
from datetime import timedelta

from ._internal_utils import to_native_string
from .adapters import HTTPAdapter
from .auth import _basic_auth_str
from .compat import Mapping, cookielib, urljoin, urlparse
from .cookies import (
    RequestsCookieJar,
    cookiejar_from_dict,
    extract_cookies_to_jar,
    merge_cookies,
)
from .exceptions import (
    ChunkedEncodingError,
    ContentDecodingError,
    InvalidSchema,
    TooManyRedirects,
)
from .hooks import default_hooks, dispatch_hook

# formerly defined here, reexposed here for backward compatibility
from .models import (  # noqa: F401
    DEFAULT_REDIRECT_LIMIT,
    REDIRECT_STATI,
    PreparedRequest,
    Request,
)
from .status_codes import codes
from .structures import CaseInsensitiveDict
from .utils import (  # noqa: F401
    DEFAULT_PORTS,
    default_headers,
    get_auth_from_url,
    get_environ_proxies,
    get_netrc_auth,
    requote_uri,
    resolve_proxies,
    rewind_body,
    should_bypass_proxies,
    to_key_val_list,
)

# Preferred clock, based on which one is more accurate on a given system.
if sys.platform == "win32":
    preferred_clock = time.perf_counter
else:
    preferred_clock = time.time


def merge_setting(request_setting, session_setting, dict_class=OrderedDict):
    """Determines appropriate setting for a given request, taking into account
    the explicit setting on that request, and the setting in the session. If a
    setting is a dictionary, they will be merged together using `dict_class`
    """

    if session_setting is None:
        return request_setting

    if request_setting is None:
        return session_setting

    # Bypass if not a dictionary (e.g. verify)
    if not (
        isinstance(session_setting, Mapping) and isinstance(request_setting, Mapping)
    ):
        return request_setting

    merged_setting = dict_class(to_key_val_list(session_setting))
    merged_setting.update(to_key_val_list(request_setting))

    # Remove keys that are set to None. Extract keys first to avoid altering
    # the dictionary during iteration.
    none_keys = [k for (k, v) in merged_setting.items() if v is None]
    for key in none_keys:
        del merged_setting[key]

    return merged_setting


def merge_hooks(request_hooks, session_hooks, dict_class=OrderedDict):
    """Properly merges both requests and session hooks.

    This is necessary because when request_hooks == {'response': []}, the
    merge breaks Session hooks entirely.
    """
    if session_hooks is None or session_hooks.get("response") == []:
        return request_hooks

    if request_hooks is None or request_hooks.get("response") == []:
        return session_hooks

    return merge_setting(request_hooks, session_hooks, dict_class)


class SessionRedirectMixin:
    def get_redirect_target(self, resp):
        """Receives a Response. Returns a redirect URI or ``None``"""
        # Due to the nature of how requests processes redirects this method will
        # be called at least once upon the original response and at least twice
        # on each subsequent redirect response (if any).
        # If a custom mixin is used to handle this logic, it may be advantageous
        # to cache the redirect location onto the response object as a private
        # attribute.
        if resp.is_redirect:
            location = resp.headers["location"]
            # Currently the underlying http module on py3 decode headers
            # in latin1, but empirical evidence suggests that latin1 is very
            # rarely used with non-ASCII characters in HTTP headers.
            # It is more likely to get UTF8 header rather than latin1.
            # This causes incorrect handling of UTF8 encoded location headers.
            # To solve this, we re-encode the location in latin1.
            location = location.encode("latin1")
            return to_native_string(location, "utf8")
        return None

    def should_strip_auth(self, old_url, new_url):
        """Decide whether Authorization header should be removed when redirecting"""
        old_parsed = urlparse(old_url)
        new_parsed = urlparse(new_url)
        if old_parsed.hostname != new_parsed.hostname:
            return True
        # Special case: allow http -> https redirect when using the standard
        # ports. This isn't specified by RFC 7235, but is kept to avoid
        # breaking backwards compatibility with older versions of requests
        # that allowed any redirects on the same host.
        if (
            old_parsed.scheme == "http"
            and old_parsed.port in (80, None)
            and new_parsed.scheme == "https"
            and new_parsed.port in (443, None)
        ):
            return False

        # Handle default port usage corresponding to scheme.
        changed_port = old_parsed.port != new_parsed.port
        changed_scheme = old_parsed.scheme != new_parsed.scheme
        default_port = (DEFAULT_PORTS.get(old_parsed.scheme, None), None)
        if (
            not changed_scheme
            and old_parsed.port in default_port
            and new_parsed.port in default_port
        ):
            return False

        # Standard case: root URI must match
        return changed_port or changed_scheme

    def resolve_redirects(
        self,
        resp,
        req,
        stream=False,
        timeout=None,
        verify=True,
        cert=None,
        proxies=None,
        yield_requests=False,
        **adapter_kwargs,
    ):
        """Receives a Response. Returns a generator of Responses or Requests."""

        hist = []  # keep track of history

        url = self.get_redirect_target(resp)
        previous_fragment = urlparse(req.url).fragment
        while url:
            prepared_request = req.copy()

            # Update history and keep track of redirects.
            # resp.history must ignore the original request in this loop
            hist.append(resp)
            resp.history = hist[1:]

            try:
                resp.content  # Consume socket so it can be released
            except (ChunkedEncodingError, ContentDecodingError, RuntimeError):
                resp.raw.read(decode_content=False)

            if len(resp.history) >= self.max_redirects:
                raise TooManyRedirects(
                    f"Exceeded {self.max_redirects} redirects.", response=resp
                )

            # Release the connection back into the pool.
            resp.close()

            # Handle redirection without scheme (see: RFC 1808 Section 4)
            if url.startswith("//"):
                parsed_rurl = urlparse(resp.url)
                url = ":".join([to_native_string(parsed_rurl.scheme), url])

            # Normalize url case and attach previous fragment if needed (RFC 7231 7.1.2)
            parsed = urlparse(url)
            if parsed.fragment == "" and previous_fragment:
                parsed = parsed._replace(fragment=previous_fragment)
            elif parsed.fragment:
                previous_fragment = parsed.fragment
            url = parsed.geturl()

            # Facilitate relative 'location' headers, as allowed by RFC 7231.
            # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
            # Compliant with RFC3986, we percent encode the url.
            if not parsed.netloc:
                url = urljoin(resp.url, requote_uri(url))
            else:
                url = requote_uri(url)

            prepared_request.url = to_native_string(url)

            self.rebuild_method(prepared_request, resp)

            # https://github.com/psf/requests/issues/1084
            if resp.status_code not in (
                codes.temporary_redirect,
                codes.permanent_redirect,
            ):
                # https://github.com/psf/requests/issues/3490
                purged_headers = ("Content-Length", "Content-Type", "Transfer-Encoding")
                for header in purged_headers:
                    prepared_request.headers.pop(header, None)
                prepared_request.body = None

            headers = prepared_request.headers
            headers.pop("Cookie", None)

            # Extract any cookies sent on the response to the cookiejar
            # in the new request. Because we've mutated our copied prepared
            # request, use the old one that we haven't yet touched.
            extract_cookies_to_jar(prepared_request._cookies, req, resp.raw)
            merge_cookies(prepared_request._cookies, self.cookies)
            prepared_request.prepare_cookies(prepared_request._cookies)

            # Rebuild auth and proxy information.
            proxies = self.rebuild_proxies(prepared_request, proxies)
            self.rebuild_auth(prepared_request, resp)

            # A failed tell() sets `_body_position` to `object()`. This non-None
            # value ensures `rewindable` will be True, allowing us to raise an
            # UnrewindableBodyError, instead of hanging the connection.
            rewindable = prepared_request._body_position is not None and (
                "Content-Length" in headers or "Transfer-Encoding" in headers
            )

            # Attempt to rewind consumed file-like object.
            if rewindable:
                rewind_body(prepared_request)

            # Override the original request.
            req = prepared_request

            if yield_requests:
                yield req
            else:
                resp = self.send(
                    req,
                    stream=stream,
                    timeout=timeout,
                    verify=verify,
                    cert=cert,
                    proxies=proxies,
                    allow_redirects=False,
                    **adapter_kwargs,
                )

                extract_cookies_to_jar(self.cookies, prepared_request, resp.raw)

                # extract redirect url, if any, for the next loop
                url = self.get_redirect_target(resp)
                yield resp

    def rebuild_auth(self, prepared_request, response):
        """When being redirected we may want to strip authentication from the
        request to avoid leaking credentials. This method intelligently removes
        and reapplies authentication where possible to avoid credential loss.
        """
        headers = prepared_request.headers
        url = prepared_request.url

        if "Authorization" in headers and self.should_strip_auth(
            response.request.url, url
        ):
            # If we get redirected to a new host, we should strip out any
            # authentication headers.
            del headers["Authorization"]

        # .netrc might have more auth for us on our new host.
        new_auth = get_netrc_auth(url) if self.trust_env else None
        if new_auth is not None:
            prepared_request.prepare_auth(new_auth)

    def rebuild_proxies(self, prepared_request, proxies):
        """This method re-evaluates the proxy configuration by considering the
        environment variables. If we are redirected to a URL covered by
        NO_PROXY, we strip the proxy configuration. Otherwise, we set missing
        proxy keys for this URL (in case they were stripped by a previous
        redirect).

        This method also replaces the Proxy-Authorization header where
        necessary.

        :rtype: dict
        """
        headers = prepared_request.headers
        scheme = urlparse(prepared_request.url).scheme
        new_proxies = resolve_proxies(prepared_request, proxies, self.trust_env)

        if "Proxy-Authorization" in headers:
            del headers["Proxy-Authorization"]

        try:
            username, password = get_auth_from_url(new_proxies[scheme])
        except KeyError:
            username, password = None, None

        # urllib3 handles proxy authorization for us in the standard adapter.
        # Avoid appending this to TLS tunneled requests where it may be leaked.
        if not scheme.startswith("https") and username and password:
            headers["Proxy-Authorization"] = _basic_auth_str(username, password)

        return new_proxies

    def rebuild_method(self, prepared_request, response):
        """When being redirected we may want to change the method of the request
        based on certain specs or browser behavior.
        """
        method = prepared_request.method

        # https://tools.ietf.org/html/rfc7231#section-6.4.4
        if response.status_code == codes.see_other and method != "HEAD":
            method = "GET"

        # Do what the browsers do, despite standards...
        # First, turn 302s into GETs.
        if response.status_code == codes.found and method != "HEAD":
            method = "GET"

        # Second, if a POST is responded to with a 301, turn it into a GET.
        # This bizarre behaviour is explained in Issue 1704.
        if response.status_code == codes.moved and method == "POST":
            method = "GET"

        prepared_request.method = method


class Session(SessionRedirectMixin):
    """A Requests session.

    Provides cookie persistence, connection-pooling, and configuration.

    Basic Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> s.get('https://httpbin.org/get')
      <Response [200]>

    Or as a context manager::

      >>> with requests.Session() as s:
      ...     s.get('https://httpbin.org/get')
      <Response [200]>
    """

    __attrs__ = [
        "headers",
        "cookies",
        "auth",
        "proxies",
        "hooks",
        "params",
        "verify",
        "cert",
        "adapters",
        "stream",
        "trust_env",
        "max_redirects",
    ]

    def __init__(self):
        #: A case-insensitive dictionary of headers to be sent on each
        #: :class:`Request <Request>` sent from this
        #: :class:`Session <Session>`.
        self.headers = default_headers()

        #: Default Authentication tuple or object to attach to
        #: :class:`Request <Request>`.
        self.auth = None

        #: Dictionary mapping protocol or protocol and host to the URL of the proxy
        #: (e.g. {'http': 'foo.bar:3128', 'http://host.name': 'foo.bar:4012'}) to
        #: be used on each :class:`Request <Request>`.
        self.proxies = {}

        #: Event-handling hooks.
        self.hooks = default_hooks()

        #: Dictionary of querystring data to attach to each
        #: :class:`Request <Request>`. The dictionary values may be lists for
        #: representing multivalued query parameters.
        self.params = {}

        #: Stream response content default.
        self.stream = False

        #: SSL Verification default.
        #: Defaults to `True`, requiring requests to verify the TLS certificate at the
        #: remote end.
        #: If verify is set to `False`, requests will accept any TLS certificate
        #: presented by the server, and will ignore hostname mismatches and/or
        #: expired certificates, which will make your application vulnerable to
        #: man-in-the-middle (MitM) attacks.
        #: Only set this to `False` for testing.
        self.verify = True

        #: SSL client certificate default, if String, path to ssl client
        #: cert file (.pem). If Tuple, ('cert', 'key') pair.
        self.cert = None

        #: Maximum number of redirects allowed. If the request exceeds this
        #: limit, a :class:`TooManyRedirects` exception is raised.
        #: This defaults to requests.models.DEFAULT_REDIRECT_LIMIT, which is
        #: 30.
        self.max_redirects = DEFAULT_REDIRECT_LIMIT

        #: Trust environment settings for proxy configuration, default
        #: authentication and similar.
        self.trust_env = True

        #: A CookieJar containing all currently outstanding cookies set on this
        #: session. By default it is a
        #: :class:`RequestsCookieJar <requests.cookies.RequestsCookieJar>`, but
        #: may be any other ``cookielib.CookieJar`` compatible object.
        self.cookies = cookiejar_from_dict({})

        # Default connection adapters.
        self.adapters = OrderedDict()
        self.mount("https://", HTTPAdapter())
        self.mount("http://", HTTPAdapter())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def prepare_request(self, request):
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for
        transmission and returns it. The :class:`PreparedRequest` has settings
        merged from the :class:`Request <Request>` instance and those of the
        :class:`Session`.

        :param request: :class:`Request` instance to prepare with this
            session's settings.
        :rtype: requests.PreparedRequest
        """
        cookies = request.cookies or {}

        # Bootstrap CookieJar.
        if not isinstance(cookies, cookielib.CookieJar):
            cookies = cookiejar_from_dict(cookies)

        # Merge with session cookies
        merged_cookies = merge_cookies(
            merge_cookies(RequestsCookieJar(), self.cookies), cookies
        )

        # Set environment's basic authentication if not explicitly set.
        auth = request.auth
        if self.trust_env and not auth and not self.auth:
            auth = get_netrc_auth(request.url)

        p = PreparedRequest()
        p.prepare(
            method=request.method.upper(),
            url=request.url,
            files=request.files,
            data=request.data,
            json=request.json,
            headers=merge_setting(
                request.headers, self.headers, dict_class=CaseInsensitiveDict
            ),
            params=merge_setting(request.params, self.params),
            auth=merge_setting(auth, self.auth),
            cookies=merged_cookies,
            hooks=merge_hooks(request.hooks, self.hooks),
        )
        return p

    def request(
        self,
        method,
        url,
        params=None,
        data=None,
        headers=None,
        cookies=None,
        files=None,
        auth=None,
        timeout=None,
        allow_redirects=True,
        proxies=None,
        hooks=None,
        stream=None,
        verify=None,
        cert=None,
        json=None,
    ):
        """Constructs a :class:`Request <Request>`, prepares it and sends it.
        Returns :class:`Response <Response>` object.

        :param method: method for the new :class:`Request` object.
        :param url: URL for the new :class:`Request` object.
        :param params: (optional) Dictionary or bytes to be sent in the query
            string for the :class:`Request`.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the
            :class:`Request`.
        :param headers: (optional) Dictionary of HTTP Headers to send with the
            :class:`Request`.
        :param cookies: (optional) Dict or CookieJar object to send with the
            :class:`Request`.
        :param files: (optional) Dictionary of ``'filename': file-like-objects``
            for multipart encoding upload.
        :param auth: (optional) Auth tuple or callable to enable
            Basic/Digest/Custom HTTP Auth.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param allow_redirects: (optional) Set to True by default.
        :type allow_redirects: bool
        :param proxies: (optional) Dictionary mapping protocol or protocol and
            hostname to the URL of the proxy.
        :param hooks: (optional) Dictionary mapping hook name to one event or
            list of events, event must be callable.
        :param stream: (optional) whether to immediately download the response
            content. Defaults to ``False``.
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use. Defaults to ``True``. When set to
            ``False``, requests will accept any TLS certificate presented by
            the server, and will ignore hostname mismatches and/or expired
            certificates, which will make your application vulnerable to
            man-in-the-middle (MitM) attacks. Setting verify to ``False``
            may be useful during local development or testing.
        :param cert: (optional) if String, path to ssl client cert file (.pem).
            If Tuple, ('cert', 'key') pair.
        :rtype: requests.Response
        """
        # Create the Request.
        req = Request(
            method=method.upper(),
            url=url,
            headers=headers,
            files=files,
            data=data or {},
            json=json,
            params=params or {},
            auth=auth,
            cookies=cookies,
            hooks=hooks,
        )
        prep = self.prepare_request(req)

        proxies = proxies or {}

        settings = self.merge_environment_settings(
            prep.url, proxies, stream, verify, cert
        )

        # Send the request.
        send_kwargs = {
            "timeout": timeout,
            "allow_redirects": allow_redirects,
        }
        send_kwargs.update(settings)
        resp = self.send(prep, **send_kwargs)

        return resp

    def get(self, url, **kwargs):
        r"""Sends a GET request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", True)
        return self.request("GET", url, **kwargs)

    def options(self, url, **kwargs):
        r"""Sends a OPTIONS request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", True)
        return self.request("OPTIONS", url, **kwargs)

    def head(self, url, **kwargs):
        r"""Sends a HEAD request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", False)
        return self.request("HEAD", url, **kwargs)

    def post(self, url, data=None, json=None, **kwargs):
        r"""Sends a POST request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("POST", url, data=data, json=json, **kwargs)

    def put(self, url, data=None, **kwargs):
        r"""Sends a PUT request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("PUT", url, data=data, **kwargs)

    def patch(self, url, data=None, **kwargs):
        r"""Sends a PATCH request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("PATCH", url, data=data, **kwargs)

    def delete(self, url, **kwargs):
        r"""Sends a DELETE request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("DELETE", url, **kwargs)

    def send(self, request, **kwargs):
        """Send a given PreparedRequest.

        :rtype: requests.Response
        """
        # Set defaults that the hooks can utilize to ensure they always have
        # the correct parameters to reproduce the previous request.
        kwargs.setdefault("stream", self.stream)
        kwargs.setdefault("verify", self.verify)
        kwargs.setdefault("cert", self.cert)
        if "proxies" not in kwargs:
            kwargs["proxies"] = resolve_proxies(request, self.proxies, self.trust_env)

        # It's possible that users might accidentally send a Request object.
        # Guard against that specific failure case.
        if isinstance(request, Request):
            raise ValueError("You can only send PreparedRequests.")

        # Set up variables needed for resolve_redirects and dispatching of hooks
        allow_redirects = kwargs.pop("allow_redirects", True)
        stream = kwargs.get("stream")
        hooks = request.hooks

        # Get the appropriate adapter to use
        adapter = self.get_adapter(url=request.url)

        # Start time (approximately) of the request
        start = preferred_clock()

        # Send the request
        r = adapter.send(request, **kwargs)

        # Total elapsed time of the request (approximately)
        elapsed = preferred_clock() - start
        r.elapsed = timedelta(seconds=elapsed)

        # Response manipulation hooks
        r = dispatch_hook("response", hooks, r, **kwargs)

        # Persist cookies
        if r.history:
            # If the hooks create history then we want those cookies too
            for resp in r.history:
                extract_cookies_to_jar(self.cookies, resp.request, resp.raw)

        extract_cookies_to_jar(self.cookies, request, r.raw)

        # Resolve redirects if allowed.
        if allow_redirects:
            # Redirect resolving generator.
            gen = self.resolve_redirects(r, request, **kwargs)
            history = [resp for resp in gen]
        else:
            history = []

        # Shuffle things around if there's history.
        if history:
            # Insert the first (original) request at the start
            history.insert(0, r)
            # Get the last request made
            r = history.pop()
            r.history = history

        # If redirects aren't being followed, store the response on the Request for Response.next().
        if not allow_redirects:
            try:
                r._next = next(
                    self.resolve_redirects(r, request, yield_requests=True, **kwargs)
                )
            except StopIteration:
                pass

        if not stream:
            r.content

        return r

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        """
        Check the environment and merge it with some settings.

        :rtype: dict
        """
        # Gather clues from the surrounding environment.
        if self.trust_env:
            # Set environment's proxies.
            no_proxy = proxies.get("no_proxy") if proxies is not None else None
            env_proxies = get_environ_proxies(url, no_proxy=no_proxy)
            for k, v in env_proxies.items():
                proxies.setdefault(k, v)

            # Look for requests environment configuration
            # and be compatible with cURL.
            if verify is True or verify is None:
                verify = (
                    os.environ.get("REQUESTS_CA_BUNDLE")
                    or os.environ.get("CURL_CA_BUNDLE")
                    or verify
                )

        # Merge all the kwargs.
        proxies = merge_setting(proxies, self.proxies)
        stream = merge_setting(stream, self.stream)
        verify = merge_setting(verify, self.verify)
        cert = merge_setting(cert, self.cert)

        return {"proxies": proxies, "stream": stream, "verify": verify, "cert": cert}

    def get_adapter(self, url):
        """
        Returns the appropriate connection adapter for the given URL.

        :rtype: requests.adapters.BaseAdapter
        """
        for prefix, adapter in self.adapters.items():
            if url.lower().startswith(prefix.lower()):
                return adapter

        # Nothing matches :-/
        raise InvalidSchema(f"No connection adapters were found for {url!r}")

    def close(self):
        """Closes all adapters and as such the session"""
        for v in self.adapters.values():
            v.close()

    def mount(self, prefix, adapter):
        """Registers a connection adapter to a prefix.

        Adapters are sorted in descending order by prefix length.
        """
        self.adapters[prefix] = adapter
        keys_to_move = [k for k in self.adapters if len(k) < len(prefix)]

        for key in keys_to_move:
            self.adapters[key] = self.adapters.pop(key)

    def __getstate__(self):
        state = {attr: getattr(self, attr, None) for attr in self.__attrs__}
        return state

    def __setstate__(self, state):
        for attr, value in state.items():
            setattr(self, attr, value)


def session():
    """
    Returns a :class:`Session` for context-management.

    .. deprecated:: 1.0.0

        This method has been deprecated since version 1.0.0 and is only kept for
        backwards compatibility. New code should use :class:`~requests.sessions.Session`
        to create a session. This may be removed at a future date.

    :rtype: Session
    """
    return Session()


# === src/requests/hooks.py ===
"""
requests.hooks
~~~~~~~~~~~~~~

This module provides the capabilities for the Requests hooks system.

Available hooks:

``response``:
    The response generated from a Request.
"""
HOOKS = ["response"]


def default_hooks():
    return {event: [] for event in HOOKS}


# TODO: response is the only one


def dispatch_hook(key, hooks, hook_data, **kwargs):
    """Dispatches a hook dictionary on a given piece of data."""
    hooks = hooks or {}
    hooks = hooks.get(key)
    if hooks:
        if hasattr(hooks, "__call__"):
            hooks = [hooks]
        for hook in hooks:
            _hook_data = hook(hook_data, **kwargs)
            if _hook_data is not None:
                hook_data = _hook_data
    return hook_data


# === src/requests/compat.py ===
"""
requests.compat
~~~~~~~~~~~~~~~

This module previously handled import compatibility issues
between Python 2 and Python 3. It remains for backwards
compatibility until the next major version.
"""

import importlib
import sys

# -------
# urllib3
# -------
from urllib3 import __version__ as urllib3_version

# Detect which major version of urllib3 is being used.
try:
    is_urllib3_1 = int(urllib3_version.split(".")[0]) == 1
except (TypeError, AttributeError):
    # If we can't discern a version, prefer old functionality.
    is_urllib3_1 = True

# -------------------
# Character Detection
# -------------------


def _resolve_char_detection():
    """Find supported character detection libraries."""
    chardet = None
    for lib in ("chardet", "charset_normalizer"):
        if chardet is None:
            try:
                chardet = importlib.import_module(lib)
            except ImportError:
                pass
    return chardet


chardet = _resolve_char_detection()

# -------
# Pythons
# -------

# Syntax sugar.
_ver = sys.version_info

#: Python 2.x?
is_py2 = _ver[0] == 2

#: Python 3.x?
is_py3 = _ver[0] == 3

# json/simplejson module import resolution
has_simplejson = False
try:
    import simplejson as json

    has_simplejson = True
except ImportError:
    import json

if has_simplejson:
    from simplejson import JSONDecodeError
else:
    from json import JSONDecodeError

# Keep OrderedDict for backwards compatibility.
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from http import cookiejar as cookielib
from http.cookies import Morsel
from io import StringIO

# --------------
# Legacy Imports
# --------------
from urllib.parse import (
    quote,
    quote_plus,
    unquote,
    unquote_plus,
    urldefrag,
    urlencode,
    urljoin,
    urlparse,
    urlsplit,
    urlunparse,
)
from urllib.request import (
    getproxies,
    getproxies_environment,
    parse_http_list,
    proxy_bypass,
    proxy_bypass_environment,
)

builtin_str = str
str = str
bytes = bytes
basestring = (str, bytes)
numeric_types = (int, float)
integer_types = (int,)


# === src/requests/models.py ===
"""
requests.models
~~~~~~~~~~~~~~~

This module contains the primary objects that power Requests.
"""

import datetime

# Import encoding now, to avoid implicit import later.
# Implicit import within threads may cause LookupError when standard library is in a ZIP,
# such as in Embedded Python. See https://github.com/psf/requests/issues/3578.
import encodings.idna  # noqa: F401
from io import UnsupportedOperation

from urllib3.exceptions import (
    DecodeError,
    LocationParseError,
    ProtocolError,
    ReadTimeoutError,
    SSLError,
)
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata
from urllib3.util import parse_url

from ._internal_utils import to_native_string, unicode_is_ascii
from .auth import HTTPBasicAuth
from .compat import (
    Callable,
    JSONDecodeError,
    Mapping,
    basestring,
    builtin_str,
    chardet,
    cookielib,
)
from .compat import json as complexjson
from .compat import urlencode, urlsplit, urlunparse
from .cookies import _copy_cookie_jar, cookiejar_from_dict, get_cookie_header
from .exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    ContentDecodingError,
    HTTPError,
    InvalidJSONError,
    InvalidURL,
)
from .exceptions import JSONDecodeError as RequestsJSONDecodeError
from .exceptions import MissingSchema
from .exceptions import SSLError as RequestsSSLError
from .exceptions import StreamConsumedError
from .hooks import default_hooks
from .status_codes import codes
from .structures import CaseInsensitiveDict
from .utils import (
    check_header_validity,
    get_auth_from_url,
    guess_filename,
    guess_json_utf,
    iter_slices,
    parse_header_links,
    requote_uri,
    stream_decode_response_unicode,
    super_len,
    to_key_val_list,
)

#: The set of HTTP status codes that indicate an automatically
#: processable redirect.
REDIRECT_STATI = (
    codes.moved,  # 301
    codes.found,  # 302
    codes.other,  # 303
    codes.temporary_redirect,  # 307
    codes.permanent_redirect,  # 308
)

DEFAULT_REDIRECT_LIMIT = 30
CONTENT_CHUNK_SIZE = 10 * 1024
ITER_CHUNK_SIZE = 512


class RequestEncodingMixin:
    @property
    def path_url(self):
        """Build the path URL to use."""

        url = []

        p = urlsplit(self.url)

        path = p.path
        if not path:
            path = "/"

        url.append(path)

        query = p.query
        if query:
            url.append("?")
            url.append(query)

        return "".join(url)

    @staticmethod
    def _encode_params(data):
        """Encode parameters in a piece of data.

        Will successfully encode parameters when passed as a dict or a list of
        2-tuples. Order is retained if data is a list of 2-tuples but arbitrary
        if parameters are supplied as a dict.
        """

        if isinstance(data, (str, bytes)):
            return data
        elif hasattr(data, "read"):
            return data
        elif hasattr(data, "__iter__"):
            result = []
            for k, vs in to_key_val_list(data):
                if isinstance(vs, basestring) or not hasattr(vs, "__iter__"):
                    vs = [vs]
                for v in vs:
                    if v is not None:
                        result.append(
                            (
                                k.encode("utf-8") if isinstance(k, str) else k,
                                v.encode("utf-8") if isinstance(v, str) else v,
                            )
                        )
            return urlencode(result, doseq=True)
        else:
            return data

    @staticmethod
    def _encode_files(files, data):
        """Build the body for a multipart/form-data request.

        Will successfully encode files when passed as a dict or a list of
        tuples. Order is retained if data is a list of tuples but arbitrary
        if parameters are supplied as a dict.
        The tuples may be 2-tuples (filename, fileobj), 3-tuples (filename, fileobj, contentype)
        or 4-tuples (filename, fileobj, contentype, custom_headers).
        """
        if not files:
            raise ValueError("Files must be provided.")
        elif isinstance(data, basestring):
            raise ValueError("Data must not be a string.")

        new_fields = []
        fields = to_key_val_list(data or {})
        files = to_key_val_list(files or {})

        for field, val in fields:
            if isinstance(val, basestring) or not hasattr(val, "__iter__"):
                val = [val]
            for v in val:
                if v is not None:
                    # Don't call str() on bytestrings: in Py3 it all goes wrong.
                    if not isinstance(v, bytes):
                        v = str(v)

                    new_fields.append(
                        (
                            field.decode("utf-8")
                            if isinstance(field, bytes)
                            else field,
                            v.encode("utf-8") if isinstance(v, str) else v,
                        )
                    )

        for k, v in files:
            # support for explicit filename
            ft = None
            fh = None
            if isinstance(v, (tuple, list)):
                if len(v) == 2:
                    fn, fp = v
                elif len(v) == 3:
                    fn, fp, ft = v
                else:
                    fn, fp, ft, fh = v
            else:
                fn = guess_filename(v) or k
                fp = v

            if isinstance(fp, (str, bytes, bytearray)):
                fdata = fp
            elif hasattr(fp, "read"):
                fdata = fp.read()
            elif fp is None:
                continue
            else:
                fdata = fp

            rf = RequestField(name=k, data=fdata, filename=fn, headers=fh)
            rf.make_multipart(content_type=ft)
            new_fields.append(rf)

        body, content_type = encode_multipart_formdata(new_fields)

        return body, content_type


class RequestHooksMixin:
    def register_hook(self, event, hook):
        """Properly register a hook."""

        if event not in self.hooks:
            raise ValueError(f'Unsupported event specified, with event name "{event}"')

        if isinstance(hook, Callable):
            self.hooks[event].append(hook)
        elif hasattr(hook, "__iter__"):
            self.hooks[event].extend(h for h in hook if isinstance(h, Callable))

    def deregister_hook(self, event, hook):
        """Deregister a previously registered hook.
        Returns True if the hook existed, False if not.
        """

        try:
            self.hooks[event].remove(hook)
            return True
        except ValueError:
            return False


class Request(RequestHooksMixin):
    """A user-created :class:`Request <Request>` object.

    Used to prepare a :class:`PreparedRequest <PreparedRequest>`, which is sent to the server.

    :param method: HTTP method to use.
    :param url: URL to send.
    :param headers: dictionary of headers to send.
    :param files: dictionary of {filename: fileobject} files to multipart upload.
    :param data: the body to attach to the request. If a dictionary or
        list of tuples ``[(key, value)]`` is provided, form-encoding will
        take place.
    :param json: json for the body to attach to the request (if files or data is not specified).
    :param params: URL parameters to append to the URL. If a dictionary or
        list of tuples ``[(key, value)]`` is provided, form-encoding will
        take place.
    :param auth: Auth handler or (user, pass) tuple.
    :param cookies: dictionary or CookieJar of cookies to attach to this request.
    :param hooks: dictionary of callback hooks, for internal usage.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'https://httpbin.org/get')
      >>> req.prepare()
      <PreparedRequest [GET]>
    """

    def __init__(
        self,
        method=None,
        url=None,
        headers=None,
        files=None,
        data=None,
        params=None,
        auth=None,
        cookies=None,
        hooks=None,
        json=None,
    ):
        # Default empty dicts for dict params.
        data = [] if data is None else data
        files = [] if files is None else files
        headers = {} if headers is None else headers
        params = {} if params is None else params
        hooks = {} if hooks is None else hooks

        self.hooks = default_hooks()
        for k, v in list(hooks.items()):
            self.register_hook(event=k, hook=v)

        self.method = method
        self.url = url
        self.headers = headers
        self.files = files
        self.data = data
        self.json = json
        self.params = params
        self.auth = auth
        self.cookies = cookies

    def __repr__(self):
        return f"<Request [{self.method}]>"

    def prepare(self):
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for transmission and returns it."""
        p = PreparedRequest()
        p.prepare(
            method=self.method,
            url=self.url,
            headers=self.headers,
            files=self.files,
            data=self.data,
            json=self.json,
            params=self.params,
            auth=self.auth,
            cookies=self.cookies,
            hooks=self.hooks,
        )
        return p


class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):
    """The fully mutable :class:`PreparedRequest <PreparedRequest>` object,
    containing the exact bytes that will be sent to the server.

    Instances are generated from a :class:`Request <Request>` object, and
    should not be instantiated manually; doing so may produce undesirable
    effects.

    Usage::

      >>> import requests
      >>> req = requests.Request('GET', 'https://httpbin.org/get')
      >>> r = req.prepare()
      >>> r
      <PreparedRequest [GET]>

      >>> s = requests.Session()
      >>> s.send(r)
      <Response [200]>
    """

    def __init__(self):
        #: HTTP verb to send to the server.
        self.method = None
        #: HTTP URL to send the request to.
        self.url = None
        #: dictionary of HTTP headers.
        self.headers = None
        # The `CookieJar` used to create the Cookie header will be stored here
        # after prepare_cookies is called
        self._cookies = None
        #: request body to send to the server.
        self.body = None
        #: dictionary of callback hooks, for internal usage.
        self.hooks = default_hooks()
        #: integer denoting starting position of a readable file-like body.
        self._body_position = None

    def prepare(
        self,
        method=None,
        url=None,
        headers=None,
        files=None,
        data=None,
        params=None,
        auth=None,
        cookies=None,
        hooks=None,
        json=None,
    ):
        """Prepares the entire request with the given parameters."""

        self.prepare_method(method)
        self.prepare_url(url, params)
        self.prepare_headers(headers)
        self.prepare_cookies(cookies)
        self.prepare_body(data, files, json)
        self.prepare_auth(auth, url)

        # Note that prepare_auth must be last to enable authentication schemes
        # such as OAuth to work on a fully prepared request.

        # This MUST go after prepare_auth. Authenticators could add a hook
        self.prepare_hooks(hooks)

    def __repr__(self):
        return f"<PreparedRequest [{self.method}]>"

    def copy(self):
        p = PreparedRequest()
        p.method = self.method
        p.url = self.url
        p.headers = self.headers.copy() if self.headers is not None else None
        p._cookies = _copy_cookie_jar(self._cookies)
        p.body = self.body
        p.hooks = self.hooks
        p._body_position = self._body_position
        return p

    def prepare_method(self, method):
        """Prepares the given HTTP method."""
        self.method = method
        if self.method is not None:
            self.method = to_native_string(self.method.upper())

    @staticmethod
    def _get_idna_encoded_host(host):
        import idna

        try:
            host = idna.encode(host, uts46=True).decode("utf-8")
        except idna.IDNAError:
            raise UnicodeError
        return host

    def prepare_url(self, url, params):
        """Prepares the given HTTP URL."""
        #: Accept objects that have string representations.
        #: We're unable to blindly call unicode/str functions
        #: as this will include the bytestring indicator (b'')
        #: on python 3.x.
        #: https://github.com/psf/requests/pull/2238
        if isinstance(url, bytes):
            url = url.decode("utf8")
        else:
            url = str(url)

        # Remove leading whitespaces from url
        url = url.lstrip()

        # Don't do any URL preparation for non-HTTP schemes like `mailto`,
        # `data` etc to work around exceptions from `url_parse`, which
        # handles RFC 3986 only.
        if ":" in url and not url.lower().startswith("http"):
            self.url = url
            return

        # Support for unicode domain names and paths.
        try:
            scheme, auth, host, port, path, query, fragment = parse_url(url)
        except LocationParseError as e:
            raise InvalidURL(*e.args)

        if not scheme:
            raise MissingSchema(
                f"Invalid URL {url!r}: No scheme supplied. "
                f"Perhaps you meant https://{url}?"
            )

        if not host:
            raise InvalidURL(f"Invalid URL {url!r}: No host supplied")

        # In general, we want to try IDNA encoding the hostname if the string contains
        # non-ASCII characters. This allows users to automatically get the correct IDNA
        # behaviour. For strings containing only ASCII characters, we need to also verify
        # it doesn't start with a wildcard (*), before allowing the unencoded hostname.
        if not unicode_is_ascii(host):
            try:
                host = self._get_idna_encoded_host(host)
            except UnicodeError:
                raise InvalidURL("URL has an invalid label.")
        elif host.startswith(("*", ".")):
            raise InvalidURL("URL has an invalid label.")

        # Carefully reconstruct the network location
        netloc = auth or ""
        if netloc:
            netloc += "@"
        netloc += host
        if port:
            netloc += f":{port}"

        # Bare domains aren't valid URLs.
        if not path:
            path = "/"

        if isinstance(params, (str, bytes)):
            params = to_native_string(params)

        enc_params = self._encode_params(params)
        if enc_params:
            if query:
                query = f"{query}&{enc_params}"
            else:
                query = enc_params

        url = requote_uri(urlunparse([scheme, netloc, path, None, query, fragment]))
        self.url = url

    def prepare_headers(self, headers):
        """Prepares the given HTTP headers."""

        self.headers = CaseInsensitiveDict()
        if headers:
            for header in headers.items():
                # Raise exception on invalid header value.
                check_header_validity(header)
                name, value = header
                self.headers[to_native_string(name)] = value

    def prepare_body(self, data, files, json=None):
        """Prepares the given HTTP body data."""

        # Check if file, fo, generator, iterator.
        # If not, run through normal process.

        # Nottin' on you.
        body = None
        content_type = None

        if not data and json is not None:
            # urllib3 requires a bytes-like body. Python 2's json.dumps
            # provides this natively, but Python 3 gives a Unicode string.
            content_type = "application/json"

            try:
                body = complexjson.dumps(json, allow_nan=False)
            except ValueError as ve:
                raise InvalidJSONError(ve, request=self)

            if not isinstance(body, bytes):
                body = body.encode("utf-8")

        is_stream = all(
            [
                hasattr(data, "__iter__"),
                not isinstance(data, (basestring, list, tuple, Mapping)),
            ]
        )

        if is_stream:
            try:
                length = super_len(data)
            except (TypeError, AttributeError, UnsupportedOperation):
                length = None

            body = data

            if getattr(body, "tell", None) is not None:
                # Record the current file position before reading.
                # This will allow us to rewind a file in the event
                # of a redirect.
                try:
                    self._body_position = body.tell()
                except OSError:
                    # This differentiates from None, allowing us to catch
                    # a failed `tell()` later when trying to rewind the body
                    self._body_position = object()

            if files:
                raise NotImplementedError(
                    "Streamed bodies and files are mutually exclusive."
                )

            if length:
                self.headers["Content-Length"] = builtin_str(length)
            else:
                self.headers["Transfer-Encoding"] = "chunked"
        else:
            # Multi-part file uploads.
            if files:
                (body, content_type) = self._encode_files(files, data)
            else:
                if data:
                    body = self._encode_params(data)
                    if isinstance(data, basestring) or hasattr(data, "read"):
                        content_type = None
                    else:
                        content_type = "application/x-www-form-urlencoded"

            self.prepare_content_length(body)

            # Add content-type if it wasn't explicitly provided.
            if content_type and ("content-type" not in self.headers):
                self.headers["Content-Type"] = content_type

        self.body = body

    def prepare_content_length(self, body):
        """Prepare Content-Length header based on request method and body"""
        if body is not None:
            length = super_len(body)
            if length:
                # If length exists, set it. Otherwise, we fallback
                # to Transfer-Encoding: chunked.
                self.headers["Content-Length"] = builtin_str(length)
        elif (
            self.method not in ("GET", "HEAD")
            and self.headers.get("Content-Length") is None
        ):
            # Set Content-Length to 0 for methods that can have a body
            # but don't provide one. (i.e. not GET or HEAD)
            self.headers["Content-Length"] = "0"

    def prepare_auth(self, auth, url=""):
        """Prepares the given HTTP auth data."""

        # If no Auth is explicitly provided, extract it from the URL first.
        if auth is None:
            url_auth = get_auth_from_url(self.url)
            auth = url_auth if any(url_auth) else None

        if auth:
            if isinstance(auth, tuple) and len(auth) == 2:
                # special-case basic HTTP auth
                auth = HTTPBasicAuth(*auth)

            # Allow auth to make its changes.
            r = auth(self)

            # Update self to reflect the auth changes.
            self.__dict__.update(r.__dict__)

            # Recompute Content-Length
            self.prepare_content_length(self.body)

    def prepare_cookies(self, cookies):
        """Prepares the given HTTP cookie data.

        This function eventually generates a ``Cookie`` header from the
        given cookies using cookielib. Due to cookielib's design, the header
        will not be regenerated if it already exists, meaning this function
        can only be called once for the life of the
        :class:`PreparedRequest <PreparedRequest>` object. Any subsequent calls
        to ``prepare_cookies`` will have no actual effect, unless the "Cookie"
        header is removed beforehand.
        """
        if isinstance(cookies, cookielib.CookieJar):
            self._cookies = cookies
        else:
            self._cookies = cookiejar_from_dict(cookies)

        cookie_header = get_cookie_header(self._cookies, self)
        if cookie_header is not None:
            self.headers["Cookie"] = cookie_header

    def prepare_hooks(self, hooks):
        """Prepares the given hooks."""
        # hooks can be passed as None to the prepare method and to this
        # method. To prevent iterating over None, simply use an empty list
        # if hooks is False-y
        hooks = hooks or []
        for event in hooks:
            self.register_hook(event, hooks[event])


class Response:
    """The :class:`Response <Response>` object, which contains a
    server's response to an HTTP request.
    """

    __attrs__ = [
        "_content",
        "status_code",
        "headers",
        "url",
        "history",
        "encoding",
        "reason",
        "cookies",
        "elapsed",
        "request",
    ]

    def __init__(self):
        self._content = False
        self._content_consumed = False
        self._next = None

        #: Integer Code of responded HTTP Status, e.g. 404 or 200.
        self.status_code = None

        #: Case-insensitive Dictionary of Response Headers.
        #: For example, ``headers['content-encoding']`` will return the
        #: value of a ``'Content-Encoding'`` response header.
        self.headers = CaseInsensitiveDict()

        #: File-like object representation of response (for advanced usage).
        #: Use of ``raw`` requires that ``stream=True`` be set on the request.
        #: This requirement does not apply for use internally to Requests.
        self.raw = None

        #: Final URL location of Response.
        self.url = None

        #: Encoding to decode with when accessing r.text.
        self.encoding = None

        #: A list of :class:`Response <Response>` objects from
        #: the history of the Request. Any redirect responses will end
        #: up here. The list is sorted from the oldest to the most recent request.
        self.history = []

        #: Textual reason of responded HTTP Status, e.g. "Not Found" or "OK".
        self.reason = None

        #: A CookieJar of Cookies the server sent back.
        self.cookies = cookiejar_from_dict({})

        #: The amount of time elapsed between sending the request
        #: and the arrival of the response (as a timedelta).
        #: This property specifically measures the time taken between sending
        #: the first byte of the request and finishing parsing the headers. It
        #: is therefore unaffected by consuming the response content or the
        #: value of the ``stream`` keyword argument.
        self.elapsed = datetime.timedelta(0)

        #: The :class:`PreparedRequest <PreparedRequest>` object to which this
        #: is a response.
        self.request = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getstate__(self):
        # Consume everything; accessing the content attribute makes
        # sure the content has been fully read.
        if not self._content_consumed:
            self.content

        return {attr: getattr(self, attr, None) for attr in self.__attrs__}

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

        # pickled objects do not have .raw
        setattr(self, "_content_consumed", True)
        setattr(self, "raw", None)

    def __repr__(self):
        return f"<Response [{self.status_code}]>"

    def __bool__(self):
        """Returns True if :attr:`status_code` is less than 400.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code, is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        return self.ok

    def __nonzero__(self):
        """Returns True if :attr:`status_code` is less than 400.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code, is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        return self.ok

    def __iter__(self):
        """Allows you to use a response as an iterator."""
        return self.iter_content(128)

    @property
    def ok(self):
        """Returns True if :attr:`status_code` is less than 400, False if not.

        This attribute checks if the status code of the response is between
        400 and 600 to see if there was a client error or a server error. If
        the status code is between 200 and 400, this will return True. This
        is **not** a check to see if the response code is ``200 OK``.
        """
        try:
            self.raise_for_status()
        except HTTPError:
            return False
        return True

    @property
    def is_redirect(self):
        """True if this Response is a well-formed HTTP redirect that could have
        been processed automatically (by :meth:`Session.resolve_redirects`).
        """
        return "location" in self.headers and self.status_code in REDIRECT_STATI

    @property
    def is_permanent_redirect(self):
        """True if this Response one of the permanent versions of redirect."""
        return "location" in self.headers and self.status_code in (
            codes.moved_permanently,
            codes.permanent_redirect,
        )

    @property
    def next(self):
        """Returns a PreparedRequest for the next request in a redirect chain, if there is one."""
        return self._next

    @property
    def apparent_encoding(self):
        """The apparent encoding, provided by the charset_normalizer or chardet libraries."""
        if chardet is not None:
            return chardet.detect(self.content)["encoding"]
        else:
            # If no character detection library is available, we'll fall back
            # to a standard Python utf-8 str.
            return "utf-8"

    def iter_content(self, chunk_size=1, decode_unicode=False):
        """Iterates over the response data.  When stream=True is set on the
        request, this avoids reading the content at once into memory for
        large responses.  The chunk size is the number of bytes it should
        read into memory.  This is not necessarily the length of each item
        returned as decoding can take place.

        chunk_size must be of type int or None. A value of None will
        function differently depending on the value of `stream`.
        stream=True will read data as it arrives in whatever size the
        chunks are received. If stream=False, data is returned as
        a single chunk.

        If decode_unicode is True, content will be decoded using the best
        available encoding based on the response.
        """

        def generate():
            # Special case for urllib3.
            if hasattr(self.raw, "stream"):
                try:
                    yield from self.raw.stream(chunk_size, decode_content=True)
                except ProtocolError as e:
                    raise ChunkedEncodingError(e)
                except DecodeError as e:
                    raise ContentDecodingError(e)
                except ReadTimeoutError as e:
                    raise ConnectionError(e)
                except SSLError as e:
                    raise RequestsSSLError(e)
            else:
                # Standard file-like object.
                while True:
                    chunk = self.raw.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

            self._content_consumed = True

        if self._content_consumed and isinstance(self._content, bool):
            raise StreamConsumedError()
        elif chunk_size is not None and not isinstance(chunk_size, int):
            raise TypeError(
                f"chunk_size must be an int, it is instead a {type(chunk_size)}."
            )
        # simulate reading small chunks of the content
        reused_chunks = iter_slices(self._content, chunk_size)

        stream_chunks = generate()

        chunks = reused_chunks if self._content_consumed else stream_chunks

        if decode_unicode:
            chunks = stream_decode_response_unicode(chunks, self)

        return chunks

    def iter_lines(
        self, chunk_size=ITER_CHUNK_SIZE, decode_unicode=False, delimiter=None
    ):
        """Iterates over the response data, one line at a time.  When
        stream=True is set on the request, this avoids reading the
        content at once into memory for large responses.

        .. note:: This method is not reentrant safe.
        """

        pending = None

        for chunk in self.iter_content(
            chunk_size=chunk_size, decode_unicode=decode_unicode
        ):
            if pending is not None:
                chunk = pending + chunk

            if delimiter:
                lines = chunk.split(delimiter)
            else:
                lines = chunk.splitlines()

            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None

            yield from lines

        if pending is not None:
            yield pending

    @property
    def content(self):
        """Content of the response, in bytes."""

        if self._content is False:
            # Read the contents.
            if self._content_consumed:
                raise RuntimeError("The content for this response was already consumed")

            if self.status_code == 0 or self.raw is None:
                self._content = None
            else:
                self._content = b"".join(self.iter_content(CONTENT_CHUNK_SIZE)) or b""

        self._content_consumed = True
        # don't need to release the connection; that's been handled by urllib3
        # since we exhausted the data.
        return self._content

    @property
    def text(self):
        """Content of the response, in unicode.

        If Response.encoding is None, encoding will be guessed using
        ``charset_normalizer`` or ``chardet``.

        The encoding of the response content is determined based solely on HTTP
        headers, following RFC 2616 to the letter. If you can take advantage of
        non-HTTP knowledge to make a better guess at the encoding, you should
        set ``r.encoding`` appropriately before accessing this property.
        """

        # Try charset from content-type
        content = None
        encoding = self.encoding

        if not self.content:
            return ""

        # Fallback to auto-detected encoding.
        if self.encoding is None:
            encoding = self.apparent_encoding

        # Decode unicode from given encoding.
        try:
            content = str(self.content, encoding, errors="replace")
        except (LookupError, TypeError):
            # A LookupError is raised if the encoding was not found which could
            # indicate a misspelling or similar mistake.
            #
            # A TypeError can be raised if encoding is None
            #
            # So we try blindly encoding.
            content = str(self.content, errors="replace")

        return content

    def json(self, **kwargs):
        r"""Decodes the JSON response body (if any) as a Python object.

        This may return a dictionary, list, etc. depending on what is in the response.

        :param \*\*kwargs: Optional arguments that ``json.loads`` takes.
        :raises requests.exceptions.JSONDecodeError: If the response body does not
            contain valid json.
        """

        if not self.encoding and self.content and len(self.content) > 3:
            # No encoding set. JSON RFC 4627 section 3 states we should expect
            # UTF-8, -16 or -32. Detect which one to use; If the detection or
            # decoding fails, fall back to `self.text` (using charset_normalizer to make
            # a best guess).
            encoding = guess_json_utf(self.content)
            if encoding is not None:
                try:
                    return complexjson.loads(self.content.decode(encoding), **kwargs)
                except UnicodeDecodeError:
                    # Wrong UTF codec detected; usually because it's not UTF-8
                    # but some other 8-bit codec.  This is an RFC violation,
                    # and the server didn't bother to tell us what codec *was*
                    # used.
                    pass
                except JSONDecodeError as e:
                    raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)

        try:
            return complexjson.loads(self.text, **kwargs)
        except JSONDecodeError as e:
            # Catch JSON-related errors and raise as requests.JSONDecodeError
            # This aliases json.JSONDecodeError and simplejson.JSONDecodeError
            raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)

    @property
    def links(self):
        """Returns the parsed header links of the response, if any."""

        header = self.headers.get("link")

        resolved_links = {}

        if header:
            links = parse_header_links(header)

            for link in links:
                key = link.get("rel") or link.get("url")
                resolved_links[key] = link

        return resolved_links

    def raise_for_status(self):
        """Raises :class:`HTTPError`, if one occurred."""

        http_error_msg = ""
        if isinstance(self.reason, bytes):
            # We attempt to decode utf-8 first because some servers
            # choose to localize their reason strings. If the string
            # isn't utf-8, we fall back to iso-8859-1 for all other
            # encodings. (See PR #3538)
            try:
                reason = self.reason.decode("utf-8")
            except UnicodeDecodeError:
                reason = self.reason.decode("iso-8859-1")
        else:
            reason = self.reason

        if 400 <= self.status_code < 500:
            http_error_msg = (
                f"{self.status_code} Client Error: {reason} for url: {self.url}"
            )

        elif 500 <= self.status_code < 600:
            http_error_msg = (
                f"{self.status_code} Server Error: {reason} for url: {self.url}"
            )

        if http_error_msg:
            raise HTTPError(http_error_msg, response=self)

    def close(self):
        """Releases the connection back to the pool. Once this method has been
        called the underlying ``raw`` object must not be accessed again.

        *Note: Should not normally need to be called explicitly.*
        """
        if not self._content_consumed:
            self.raw.close()

        release_conn = getattr(self.raw, "release_conn", None)
        if release_conn is not None:
            release_conn()


# === src/requests/certs.py ===
#!/usr/bin/env python

"""
requests.certs
~~~~~~~~~~~~~~

This module returns the preferred default CA certificate bundle. There is
only one — the one from the certifi package.

If you are packaging Requests, e.g., for a Linux distribution or a managed
environment, you can change the definition of where() to return a separately
packaged CA bundle.
"""
from certifi import where

if __name__ == "__main__":
    print(where())


# === src/requests/__init__.py ===
#   __
#  /__)  _  _     _   _ _/   _
# / (   (- (/ (/ (- _)  /  _)
#          /

"""
Requests HTTP Library
~~~~~~~~~~~~~~~~~~~~~

Requests is an HTTP library, written in Python, for human beings.
Basic GET usage:

   >>> import requests
   >>> r = requests.get('https://www.python.org')
   >>> r.status_code
   200
   >>> b'Python is a programming language' in r.content
   True

... or POST:

   >>> payload = dict(key1='value1', key2='value2')
   >>> r = requests.post('https://httpbin.org/post', data=payload)
   >>> print(r.text)
   {
     ...
     "form": {
       "key1": "value1",
       "key2": "value2"
     },
     ...
   }

The other HTTP methods are supported - see `requests.api`. Full documentation
is at <https://requests.readthedocs.io>.

:copyright: (c) 2017 by Kenneth Reitz.
:license: Apache 2.0, see LICENSE for more details.
"""

import warnings

import urllib3

from .exceptions import RequestsDependencyWarning

try:
    from charset_normalizer import __version__ as charset_normalizer_version
except ImportError:
    charset_normalizer_version = None

try:
    from chardet import __version__ as chardet_version
except ImportError:
    chardet_version = None


def check_compatibility(urllib3_version, chardet_version, charset_normalizer_version):
    urllib3_version = urllib3_version.split(".")
    assert urllib3_version != ["dev"]  # Verify urllib3 isn't installed from git.

    # Sometimes, urllib3 only reports its version as 16.1.
    if len(urllib3_version) == 2:
        urllib3_version.append("0")

    # Check urllib3 for compatibility.
    major, minor, patch = urllib3_version  # noqa: F811
    major, minor, patch = int(major), int(minor), int(patch)
    # urllib3 >= 1.21.1
    assert major >= 1
    if major == 1:
        assert minor >= 21

    # Check charset_normalizer for compatibility.
    if chardet_version:
        major, minor, patch = chardet_version.split(".")[:3]
        major, minor, patch = int(major), int(minor), int(patch)
        # chardet_version >= 3.0.2, < 6.0.0
        assert (3, 0, 2) <= (major, minor, patch) < (6, 0, 0)
    elif charset_normalizer_version:
        major, minor, patch = charset_normalizer_version.split(".")[:3]
        major, minor, patch = int(major), int(minor), int(patch)
        # charset_normalizer >= 2.0.0 < 4.0.0
        assert (2, 0, 0) <= (major, minor, patch) < (4, 0, 0)
    else:
        warnings.warn(
            "Unable to find acceptable character detection dependency "
            "(chardet or charset_normalizer).",
            RequestsDependencyWarning,
        )


def _check_cryptography(cryptography_version):
    # cryptography < 1.3.4
    try:
        cryptography_version = list(map(int, cryptography_version.split(".")))
    except ValueError:
        return

    if cryptography_version < [1, 3, 4]:
        warning = "Old version of cryptography ({}) may cause slowdown.".format(
            cryptography_version
        )
        warnings.warn(warning, RequestsDependencyWarning)


# Check imported dependencies for compatibility.
try:
    check_compatibility(
        urllib3.__version__, chardet_version, charset_normalizer_version
    )
except (AssertionError, ValueError):
    warnings.warn(
        "urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
        "version!".format(
            urllib3.__version__, chardet_version, charset_normalizer_version
        ),
        RequestsDependencyWarning,
    )

# Attempt to enable urllib3's fallback for SNI support
# if the standard library doesn't support SNI or the
# 'ssl' library isn't available.
try:
    try:
        import ssl
    except ImportError:
        ssl = None

    if not getattr(ssl, "HAS_SNI", False):
        from urllib3.contrib import pyopenssl

        pyopenssl.inject_into_urllib3()

        # Check cryptography version
        from cryptography import __version__ as cryptography_version

        _check_cryptography(cryptography_version)
except ImportError:
    pass

# urllib3's DependencyWarnings should be silenced.
from urllib3.exceptions import DependencyWarning

warnings.simplefilter("ignore", DependencyWarning)

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

from . import packages, utils
from .__version__ import (
    __author__,
    __author_email__,
    __build__,
    __cake__,
    __copyright__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
)
from .api import delete, get, head, options, patch, post, put, request
from .exceptions import (
    ConnectionError,
    ConnectTimeout,
    FileModeWarning,
    HTTPError,
    JSONDecodeError,
    ReadTimeout,
    RequestException,
    Timeout,
    TooManyRedirects,
    URLRequired,
)
from .models import PreparedRequest, Request, Response
from .sessions import Session, session
from .status_codes import codes

logging.getLogger(__name__).addHandler(NullHandler())

# FileModeWarnings go off per the default.
warnings.simplefilter("default", FileModeWarning, append=True)


# === src/requests/status_codes.py ===
r"""
The ``codes`` object defines a mapping from common names for HTTP statuses
to their numerical codes, accessible either as attributes or as dictionary
items.

Example::

    >>> import requests
    >>> requests.codes['temporary_redirect']
    307
    >>> requests.codes.teapot
    418
    >>> requests.codes['\o/']
    200

Some codes have multiple names, and both upper- and lower-case versions of
the names are allowed. For example, ``codes.ok``, ``codes.OK``, and
``codes.okay`` all correspond to the HTTP status code 200.
"""

from .structures import LookupDict

_codes = {
    # Informational.
    100: ("continue",),
    101: ("switching_protocols",),
    102: ("processing", "early-hints"),
    103: ("checkpoint",),
    122: ("uri_too_long", "request_uri_too_long"),
    200: ("ok", "okay", "all_ok", "all_okay", "all_good", "\\o/", "✓"),
    201: ("created",),
    202: ("accepted",),
    203: ("non_authoritative_info", "non_authoritative_information"),
    204: ("no_content",),
    205: ("reset_content", "reset"),
    206: ("partial_content", "partial"),
    207: ("multi_status", "multiple_status", "multi_stati", "multiple_stati"),
    208: ("already_reported",),
    226: ("im_used",),
    # Redirection.
    300: ("multiple_choices",),
    301: ("moved_permanently", "moved", "\\o-"),
    302: ("found",),
    303: ("see_other", "other"),
    304: ("not_modified",),
    305: ("use_proxy",),
    306: ("switch_proxy",),
    307: ("temporary_redirect", "temporary_moved", "temporary"),
    308: (
        "permanent_redirect",
        "resume_incomplete",
        "resume",
    ),  # "resume" and "resume_incomplete" to be removed in 3.0
    # Client Error.
    400: ("bad_request", "bad"),
    401: ("unauthorized",),
    402: ("payment_required", "payment"),
    403: ("forbidden",),
    404: ("not_found", "-o-"),
    405: ("method_not_allowed", "not_allowed"),
    406: ("not_acceptable",),
    407: ("proxy_authentication_required", "proxy_auth", "proxy_authentication"),
    408: ("request_timeout", "timeout"),
    409: ("conflict",),
    410: ("gone",),
    411: ("length_required",),
    412: ("precondition_failed", "precondition"),
    413: ("request_entity_too_large", "content_too_large"),
    414: ("request_uri_too_large", "uri_too_long"),
    415: ("unsupported_media_type", "unsupported_media", "media_type"),
    416: (
        "requested_range_not_satisfiable",
        "requested_range",
        "range_not_satisfiable",
    ),
    417: ("expectation_failed",),
    418: ("im_a_teapot", "teapot", "i_am_a_teapot"),
    421: ("misdirected_request",),
    422: ("unprocessable_entity", "unprocessable", "unprocessable_content"),
    423: ("locked",),
    424: ("failed_dependency", "dependency"),
    425: ("unordered_collection", "unordered", "too_early"),
    426: ("upgrade_required", "upgrade"),
    428: ("precondition_required", "precondition"),
    429: ("too_many_requests", "too_many"),
    431: ("header_fields_too_large", "fields_too_large"),
    444: ("no_response", "none"),
    449: ("retry_with", "retry"),
    450: ("blocked_by_windows_parental_controls", "parental_controls"),
    451: ("unavailable_for_legal_reasons", "legal_reasons"),
    499: ("client_closed_request",),
    # Server Error.
    500: ("internal_server_error", "server_error", "/o\\", "✗"),
    501: ("not_implemented",),
    502: ("bad_gateway",),
    503: ("service_unavailable", "unavailable"),
    504: ("gateway_timeout",),
    505: ("http_version_not_supported", "http_version"),
    506: ("variant_also_negotiates",),
    507: ("insufficient_storage",),
    509: ("bandwidth_limit_exceeded", "bandwidth"),
    510: ("not_extended",),
    511: ("network_authentication_required", "network_auth", "network_authentication"),
}

codes = LookupDict(name="status_codes")


def _init():
    for code, titles in _codes.items():
        for title in titles:
            setattr(codes, title, code)
            if not title.startswith(("\\", "/")):
                setattr(codes, title.upper(), code)

    def doc(code):
        names = ", ".join(f"``{n}``" for n in _codes[code])
        return "* %d: %s" % (code, names)

    global __doc__
    __doc__ = (
        __doc__ + "\n" + "\n".join(doc(code) for code in sorted(_codes))
        if __doc__ is not None
        else None
    )


_init()


# === src/requests/packages.py ===
import sys

from .compat import chardet

# This code exists for backwards compatibility reasons.
# I don't like it either. Just look the other way. :)

for package in ("urllib3", "idna"):
    locals()[package] = __import__(package)
    # This traversal is apparently necessary such that the identities are
    # preserved (requests.packages.urllib3.* is urllib3.*)
    for mod in list(sys.modules):
        if mod == package or mod.startswith(f"{package}."):
            sys.modules[f"requests.packages.{mod}"] = sys.modules[mod]

if chardet is not None:
    target = chardet.__name__
    for mod in list(sys.modules):
        if mod == target or mod.startswith(f"{target}."):
            imported_mod = sys.modules[mod]
            sys.modules[f"requests.packages.{mod}"] = imported_mod
            mod = mod.replace(target, "chardet")
            sys.modules[f"requests.packages.{mod}"] = imported_mod


# === src/requests/__version__.py ===
# .-. .-. .-. . . .-. .-. .-. .-.
# |(  |-  |.| | | |-  `-.  |  `-.
# ' ' `-' `-`.`-' `-' `-'  '  `-'

__title__ = "requests"
__description__ = "Python HTTP for Humans."
__url__ = "https://requests.readthedocs.io"
__version__ = "2.32.3"
__build__ = 0x023203
__author__ = "Kenneth Reitz"
__author_email__ = "me@kennethreitz.org"
__license__ = "Apache-2.0"
__copyright__ = "Copyright Kenneth Reitz"
__cake__ = "\u2728 \U0001f370 \u2728"


# === src/requests/api.py ===
"""
requests.api
~~~~~~~~~~~~

This module implements the Requests API.

:copyright: (c) 2012 by Kenneth Reitz.
:license: Apache2, see LICENSE for more details.
"""

from . import sessions


def request(method, url, **kwargs):
    """Constructs and sends a :class:`Request <Request>`.

    :param method: method for the new :class:`Request` object: ``GET``, ``OPTIONS``, ``HEAD``, ``POST``, ``PUT``, ``PATCH``, or ``DELETE``.
    :param url: URL for the new :class:`Request` object.
    :param params: (optional) Dictionary, list of tuples or bytes to send
        in the query string for the :class:`Request`.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param headers: (optional) Dictionary of HTTP Headers to send with the :class:`Request`.
    :param cookies: (optional) Dict or CookieJar object to send with the :class:`Request`.
    :param files: (optional) Dictionary of ``'name': file-like-objects`` (or ``{'name': file-tuple}``) for multipart encoding upload.
        ``file-tuple`` can be a 2-tuple ``('filename', fileobj)``, 3-tuple ``('filename', fileobj, 'content_type')``
        or a 4-tuple ``('filename', fileobj, 'content_type', custom_headers)``, where ``'content_type'`` is a string
        defining the content type of the given file and ``custom_headers`` a dict-like object containing additional headers
        to add for the file.
    :param auth: (optional) Auth tuple to enable Basic/Digest/Custom HTTP Auth.
    :param timeout: (optional) How many seconds to wait for the server to send data
        before giving up, as a float, or a :ref:`(connect timeout, read
        timeout) <timeouts>` tuple.
    :type timeout: float or tuple
    :param allow_redirects: (optional) Boolean. Enable/disable GET/OPTIONS/POST/PUT/PATCH/DELETE/HEAD redirection. Defaults to ``True``.
    :type allow_redirects: bool
    :param proxies: (optional) Dictionary mapping protocol to the URL of the proxy.
    :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use. Defaults to ``True``.
    :param stream: (optional) if ``False``, the response content will be immediately downloaded.
    :param cert: (optional) if String, path to ssl client cert file (.pem). If Tuple, ('cert', 'key') pair.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response

    Usage::

      >>> import requests
      >>> req = requests.request('GET', 'https://httpbin.org/get')
      >>> req
      <Response [200]>
    """

    # By using the 'with' statement we are sure the session is closed, thus we
    # avoid leaving sockets open which can trigger a ResourceWarning in some
    # cases, and look like a memory leak in others.
    with sessions.Session() as session:
        return session.request(method=method, url=url, **kwargs)


def get(url, params=None, **kwargs):
    r"""Sends a GET request.

    :param url: URL for the new :class:`Request` object.
    :param params: (optional) Dictionary, list of tuples or bytes to send
        in the query string for the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("get", url, params=params, **kwargs)


def options(url, **kwargs):
    r"""Sends an OPTIONS request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("options", url, **kwargs)


def head(url, **kwargs):
    r"""Sends a HEAD request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes. If
        `allow_redirects` is not provided, it will be set to `False` (as
        opposed to the default :meth:`request` behavior).
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    kwargs.setdefault("allow_redirects", False)
    return request("head", url, **kwargs)


def post(url, data=None, json=None, **kwargs):
    r"""Sends a POST request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("post", url, data=data, json=json, **kwargs)


def put(url, data=None, **kwargs):
    r"""Sends a PUT request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("put", url, data=data, **kwargs)


def patch(url, data=None, **kwargs):
    r"""Sends a PATCH request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("patch", url, data=data, **kwargs)


def delete(url, **kwargs):
    r"""Sends a DELETE request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("delete", url, **kwargs)


# === src/requests/_internal_utils.py ===
"""
requests._internal_utils
~~~~~~~~~~~~~~

Provides utility functions that are consumed internally by Requests
which depend on extremely few external helpers (such as compat)
"""
import re

from .compat import builtin_str

_VALID_HEADER_NAME_RE_BYTE = re.compile(rb"^[^:\s][^:\r\n]*$")
_VALID_HEADER_NAME_RE_STR = re.compile(r"^[^:\s][^:\r\n]*$")
_VALID_HEADER_VALUE_RE_BYTE = re.compile(rb"^\S[^\r\n]*$|^$")
_VALID_HEADER_VALUE_RE_STR = re.compile(r"^\S[^\r\n]*$|^$")

_HEADER_VALIDATORS_STR = (_VALID_HEADER_NAME_RE_STR, _VALID_HEADER_VALUE_RE_STR)
_HEADER_VALIDATORS_BYTE = (_VALID_HEADER_NAME_RE_BYTE, _VALID_HEADER_VALUE_RE_BYTE)
HEADER_VALIDATORS = {
    bytes: _HEADER_VALIDATORS_BYTE,
    str: _HEADER_VALIDATORS_STR,
}


def to_native_string(string, encoding="ascii"):
    """Given a string object, regardless of type, returns a representation of
    that string in the native string type, encoding and decoding where
    necessary. This assumes ASCII unless told otherwise.
    """
    if isinstance(string, builtin_str):
        out = string
    else:
        out = string.decode(encoding)

    return out


def unicode_is_ascii(u_string):
    """Determine if unicode string only contains ASCII characters.

    :param str u_string: unicode string to check. Must be unicode
        and not Python 2 `str`.
    :rtype: bool
    """
    assert isinstance(u_string, str)
    try:
        u_string.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


# === src/requests/utils.py ===
"""
requests.utils
~~~~~~~~~~~~~~

This module provides utility functions that are used within Requests
that are also useful for external consumption.
"""

import codecs
import contextlib
import io
import os
import re
import socket
import struct
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict

from urllib3.util import make_headers, parse_url

from . import certs
from .__version__ import __version__

# to_native_string is unused here, but imported here for backwards compatibility
from ._internal_utils import (  # noqa: F401
    _HEADER_VALIDATORS_BYTE,
    _HEADER_VALIDATORS_STR,
    HEADER_VALIDATORS,
    to_native_string,
)
from .compat import (
    Mapping,
    basestring,
    bytes,
    getproxies,
    getproxies_environment,
    integer_types,
    is_urllib3_1,
)
from .compat import parse_http_list as _parse_list_header
from .compat import (
    proxy_bypass,
    proxy_bypass_environment,
    quote,
    str,
    unquote,
    urlparse,
    urlunparse,
)
from .cookies import cookiejar_from_dict
from .exceptions import (
    FileModeWarning,
    InvalidHeader,
    InvalidURL,
    UnrewindableBodyError,
)
from .structures import CaseInsensitiveDict

NETRC_FILES = (".netrc", "_netrc")

DEFAULT_CA_BUNDLE_PATH = certs.where()

DEFAULT_PORTS = {"http": 80, "https": 443}

# Ensure that ', ' is used to preserve previous delimiter behavior.
DEFAULT_ACCEPT_ENCODING = ", ".join(
    re.split(r",\s*", make_headers(accept_encoding=True)["accept-encoding"])
)


if sys.platform == "win32":
    # provide a proxy_bypass version on Windows without DNS lookups

    def proxy_bypass_registry(host):
        try:
            import winreg
        except ImportError:
            return False

        try:
            internetSettings = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            )
            # ProxyEnable could be REG_SZ or REG_DWORD, normalizing it
            proxyEnable = int(winreg.QueryValueEx(internetSettings, "ProxyEnable")[0])
            # ProxyOverride is almost always a string
            proxyOverride = winreg.QueryValueEx(internetSettings, "ProxyOverride")[0]
        except (OSError, ValueError):
            return False
        if not proxyEnable or not proxyOverride:
            return False

        # make a check value list from the registry entry: replace the
        # '<local>' string by the localhost entry and the corresponding
        # canonical entry.
        proxyOverride = proxyOverride.split(";")
        # filter out empty strings to avoid re.match return true in the following code.
        proxyOverride = filter(None, proxyOverride)
        # now check if we match one of the registry values.
        for test in proxyOverride:
            if test == "<local>":
                if "." not in host:
                    return True
            test = test.replace(".", r"\.")  # mask dots
            test = test.replace("*", r".*")  # change glob sequence
            test = test.replace("?", r".")  # change glob char
            if re.match(test, host, re.I):
                return True
        return False

    def proxy_bypass(host):  # noqa
        """Return True, if the host should be bypassed.

        Checks proxy settings gathered from the environment, if specified,
        or the registry.
        """
        if getproxies_environment():
            return proxy_bypass_environment(host)
        else:
            return proxy_bypass_registry(host)


def dict_to_sequence(d):
    """Returns an internal sequence dictionary update."""

    if hasattr(d, "items"):
        d = d.items()

    return d


def super_len(o):
    total_length = None
    current_position = 0

    if not is_urllib3_1 and isinstance(o, str):
        # urllib3 2.x+ treats all strings as utf-8 instead
        # of latin-1 (iso-8859-1) like http.client.
        o = o.encode("utf-8")

    if hasattr(o, "__len__"):
        total_length = len(o)

    elif hasattr(o, "len"):
        total_length = o.len

    elif hasattr(o, "fileno"):
        try:
            fileno = o.fileno()
        except (io.UnsupportedOperation, AttributeError):
            # AttributeError is a surprising exception, seeing as how we've just checked
            # that `hasattr(o, 'fileno')`.  It happens for objects obtained via
            # `Tarfile.extractfile()`, per issue 5229.
            pass
        else:
            total_length = os.fstat(fileno).st_size

            # Having used fstat to determine the file length, we need to
            # confirm that this file was opened up in binary mode.
            if "b" not in o.mode:
                warnings.warn(
                    (
                        "Requests has determined the content-length for this "
                        "request using the binary size of the file: however, the "
                        "file has been opened in text mode (i.e. without the 'b' "
                        "flag in the mode). This may lead to an incorrect "
                        "content-length. In Requests 3.0, support will be removed "
                        "for files in text mode."
                    ),
                    FileModeWarning,
                )

    if hasattr(o, "tell"):
        try:
            current_position = o.tell()
        except OSError:
            # This can happen in some weird situations, such as when the file
            # is actually a special file descriptor like stdin. In this
            # instance, we don't know what the length is, so set it to zero and
            # let requests chunk it instead.
            if total_length is not None:
                current_position = total_length
        else:
            if hasattr(o, "seek") and total_length is None:
                # StringIO and BytesIO have seek but no usable fileno
                try:
                    # seek to end of file
                    o.seek(0, 2)
                    total_length = o.tell()

                    # seek back to current position to support
                    # partially read file-like objects
                    o.seek(current_position or 0)
                except OSError:
                    total_length = 0

    if total_length is None:
        total_length = 0

    return max(0, total_length - current_position)


def get_netrc_auth(url, raise_errors=False):
    """Returns the Requests tuple auth for a given url from netrc."""

    netrc_file = os.environ.get("NETRC")
    if netrc_file is not None:
        netrc_locations = (netrc_file,)
    else:
        netrc_locations = (f"~/{f}" for f in NETRC_FILES)

    try:
        from netrc import NetrcParseError, netrc

        netrc_path = None

        for f in netrc_locations:
            try:
                loc = os.path.expanduser(f)
            except KeyError:
                # os.path.expanduser can fail when $HOME is undefined and
                # getpwuid fails. See https://bugs.python.org/issue20164 &
                # https://github.com/psf/requests/issues/1846
                return

            if os.path.exists(loc):
                netrc_path = loc
                break

        # Abort early if there isn't one.
        if netrc_path is None:
            return

        ri = urlparse(url)

        # Strip port numbers from netloc. This weird `if...encode`` dance is
        # used for Python 3.2, which doesn't support unicode literals.
        splitstr = b":"
        if isinstance(url, str):
            splitstr = splitstr.decode("ascii")
        host = ri.netloc.split(splitstr)[0]

        try:
            _netrc = netrc(netrc_path).authenticators(host)
            if _netrc:
                # Return with login / password
                login_i = 0 if _netrc[0] else 1
                return (_netrc[login_i], _netrc[2])
        except (NetrcParseError, OSError):
            # If there was a parsing error or a permissions issue reading the file,
            # we'll just skip netrc auth unless explicitly asked to raise errors.
            if raise_errors:
                raise

    # App Engine hackiness.
    except (ImportError, AttributeError):
        pass


def guess_filename(obj):
    """Tries to guess the filename of the given object."""
    name = getattr(obj, "name", None)
    if name and isinstance(name, basestring) and name[0] != "<" and name[-1] != ">":
        return os.path.basename(name)


def extract_zipped_paths(path):
    """Replace nonexistent paths that look like they refer to a member of a zip
    archive with the location of an extracted copy of the target, or else
    just return the provided path unchanged.
    """
    if os.path.exists(path):
        # this is already a valid path, no need to do anything further
        return path

    # find the first valid part of the provided path and treat that as a zip archive
    # assume the rest of the path is the name of a member in the archive
    archive, member = os.path.split(path)
    while archive and not os.path.exists(archive):
        archive, prefix = os.path.split(archive)
        if not prefix:
            # If we don't check for an empty prefix after the split (in other words, archive remains unchanged after the split),
            # we _can_ end up in an infinite loop on a rare corner case affecting a small number of users
            break
        member = "/".join([prefix, member])

    if not zipfile.is_zipfile(archive):
        return path

    zip_file = zipfile.ZipFile(archive)
    if member not in zip_file.namelist():
        return path

    # we have a valid zip archive and a valid member of that archive
    tmp = tempfile.gettempdir()
    extracted_path = os.path.join(tmp, member.split("/")[-1])
    if not os.path.exists(extracted_path):
        # use read + write to avoid the creating nested folders, we only want the file, avoids mkdir racing condition
        with atomic_open(extracted_path) as file_handler:
            file_handler.write(zip_file.read(member))
    return extracted_path


@contextlib.contextmanager
def atomic_open(filename):
    """Write a file to the disk in an atomic fashion"""
    tmp_descriptor, tmp_name = tempfile.mkstemp(dir=os.path.dirname(filename))
    try:
        with os.fdopen(tmp_descriptor, "wb") as tmp_handler:
            yield tmp_handler
        os.replace(tmp_name, filename)
    except BaseException:
        os.remove(tmp_name)
        raise


def from_key_val_list(value):
    """Take an object and test to see if it can be represented as a
    dictionary. Unless it can not be represented as such, return an
    OrderedDict, e.g.,

    ::

        >>> from_key_val_list([('key', 'val')])
        OrderedDict([('key', 'val')])
        >>> from_key_val_list('string')
        Traceback (most recent call last):
        ...
        ValueError: cannot encode objects that are not 2-tuples
        >>> from_key_val_list({'key': 'val'})
        OrderedDict([('key', 'val')])

    :rtype: OrderedDict
    """
    if value is None:
        return None

    if isinstance(value, (str, bytes, bool, int)):
        raise ValueError("cannot encode objects that are not 2-tuples")

    return OrderedDict(value)


def to_key_val_list(value):
    """Take an object and test to see if it can be represented as a
    dictionary. If it can be, return a list of tuples, e.g.,

    ::

        >>> to_key_val_list([('key', 'val')])
        [('key', 'val')]
        >>> to_key_val_list({'key': 'val'})
        [('key', 'val')]
        >>> to_key_val_list('string')
        Traceback (most recent call last):
        ...
        ValueError: cannot encode objects that are not 2-tuples

    :rtype: list
    """
    if value is None:
        return None

    if isinstance(value, (str, bytes, bool, int)):
        raise ValueError("cannot encode objects that are not 2-tuples")

    if isinstance(value, Mapping):
        value = value.items()

    return list(value)


# From mitsuhiko/werkzeug (used with permission).
def parse_list_header(value):
    """Parse lists as described by RFC 2068 Section 2.

    In particular, parse comma-separated lists where the elements of
    the list may include quoted-strings.  A quoted-string could
    contain a comma.  A non-quoted string could have quotes in the
    middle.  Quotes are removed automatically after parsing.

    It basically works like :func:`parse_set_header` just that items
    may appear multiple times and case sensitivity is preserved.

    The return value is a standard :class:`list`:

    >>> parse_list_header('token, "quoted value"')
    ['token', 'quoted value']

    To create a header from the :class:`list` again, use the
    :func:`dump_header` function.

    :param value: a string with a list header.
    :return: :class:`list`
    :rtype: list
    """
    result = []
    for item in _parse_list_header(value):
        if item[:1] == item[-1:] == '"':
            item = unquote_header_value(item[1:-1])
        result.append(item)
    return result


# From mitsuhiko/werkzeug (used with permission).
def parse_dict_header(value):
    """Parse lists of key, value pairs as described by RFC 2068 Section 2 and
    convert them into a python dict:

    >>> d = parse_dict_header('foo="is a fish", bar="as well"')
    >>> type(d) is dict
    True
    >>> sorted(d.items())
    [('bar', 'as well'), ('foo', 'is a fish')]

    If there is no value for a key it will be `None`:

    >>> parse_dict_header('key_without_value')
    {'key_without_value': None}

    To create a header from the :class:`dict` again, use the
    :func:`dump_header` function.

    :param value: a string with a dict header.
    :return: :class:`dict`
    :rtype: dict
    """
    result = {}
    for item in _parse_list_header(value):
        if "=" not in item:
            result[item] = None
            continue
        name, value = item.split("=", 1)
        if value[:1] == value[-1:] == '"':
            value = unquote_header_value(value[1:-1])
        result[name] = value
    return result


# From mitsuhiko/werkzeug (used with permission).
def unquote_header_value(value, is_filename=False):
    r"""Unquotes a header value.  (Reversal of :func:`quote_header_value`).
    This does not use the real unquoting but what browsers are actually
    using for quoting.

    :param value: the header value to unquote.
    :rtype: str
    """
    if value and value[0] == value[-1] == '"':
        # this is not the real unquoting, but fixing this so that the
        # RFC is met will result in bugs with internet explorer and
        # probably some other browsers as well.  IE for example is
        # uploading files with "C:\foo\bar.txt" as filename
        value = value[1:-1]

        # if this is a filename and the starting characters look like
        # a UNC path, then just return the value without quotes.  Using the
        # replace sequence below on a UNC path has the effect of turning
        # the leading double slash into a single slash and then
        # _fix_ie_filename() doesn't work correctly.  See #458.
        if not is_filename or value[:2] != "\\\\":
            return value.replace("\\\\", "\\").replace('\\"', '"')
    return value


def dict_from_cookiejar(cj):
    """Returns a key/value dictionary from a CookieJar.

    :param cj: CookieJar object to extract cookies from.
    :rtype: dict
    """

    cookie_dict = {cookie.name: cookie.value for cookie in cj}
    return cookie_dict


def add_dict_to_cookiejar(cj, cookie_dict):
    """Returns a CookieJar from a key/value dictionary.

    :param cj: CookieJar to insert cookies into.
    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :rtype: CookieJar
    """

    return cookiejar_from_dict(cookie_dict, cj)


def get_encodings_from_content(content):
    """Returns encodings from given content string.

    :param content: bytestring to extract encodings from.
    """
    warnings.warn(
        (
            "In requests 3.0, get_encodings_from_content will be removed. For "
            "more information, please see the discussion on issue #2266. (This"
            " warning should only appear once.)"
        ),
        DeprecationWarning,
    )

    charset_re = re.compile(r'<meta.*?charset=["\']*(.+?)["\'>]', flags=re.I)
    pragma_re = re.compile(r'<meta.*?content=["\']*;?charset=(.+?)["\'>]', flags=re.I)
    xml_re = re.compile(r'^<\?xml.*?encoding=["\']*(.+?)["\'>]')

    return (
        charset_re.findall(content)
        + pragma_re.findall(content)
        + xml_re.findall(content)
    )


def _parse_content_type_header(header):
    """Returns content type and parameters from given header

    :param header: string
    :return: tuple containing content type and dictionary of
         parameters
    """

    tokens = header.split(";")
    content_type, params = tokens[0].strip(), tokens[1:]
    params_dict = {}
    items_to_strip = "\"' "

    for param in params:
        param = param.strip()
        if param:
            key, value = param, True
            index_of_equals = param.find("=")
            if index_of_equals != -1:
                key = param[:index_of_equals].strip(items_to_strip)
                value = param[index_of_equals + 1 :].strip(items_to_strip)
            params_dict[key.lower()] = value
    return content_type, params_dict


def get_encoding_from_headers(headers):
    """Returns encodings from given HTTP Header Dict.

    :param headers: dictionary to extract encoding from.
    :rtype: str
    """

    content_type = headers.get("content-type")

    if not content_type:
        return None

    content_type, params = _parse_content_type_header(content_type)

    if "charset" in params:
        return params["charset"].strip("'\"")

    if "text" in content_type:
        return "ISO-8859-1"

    if "application/json" in content_type:
        # Assume UTF-8 based on RFC 4627: https://www.ietf.org/rfc/rfc4627.txt since the charset was unset
        return "utf-8"


def stream_decode_response_unicode(iterator, r):
    """Stream decodes an iterator."""

    if r.encoding is None:
        yield from iterator
        return

    decoder = codecs.getincrementaldecoder(r.encoding)(errors="replace")
    for chunk in iterator:
        rv = decoder.decode(chunk)
        if rv:
            yield rv
    rv = decoder.decode(b"", final=True)
    if rv:
        yield rv


def iter_slices(string, slice_length):
    """Iterate over slices of a string."""
    pos = 0
    if slice_length is None or slice_length <= 0:
        slice_length = len(string)
    while pos < len(string):
        yield string[pos : pos + slice_length]
        pos += slice_length


def get_unicode_from_response(r):
    """Returns the requested content back in unicode.

    :param r: Response object to get unicode content from.

    Tried:

    1. charset from content-type
    2. fall back and replace all unicode characters

    :rtype: str
    """
    warnings.warn(
        (
            "In requests 3.0, get_unicode_from_response will be removed. For "
            "more information, please see the discussion on issue #2266. (This"
            " warning should only appear once.)"
        ),
        DeprecationWarning,
    )

    tried_encodings = []

    # Try charset from content-type
    encoding = get_encoding_from_headers(r.headers)

    if encoding:
        try:
            return str(r.content, encoding)
        except UnicodeError:
            tried_encodings.append(encoding)

    # Fall back:
    try:
        return str(r.content, encoding, errors="replace")
    except TypeError:
        return r.content


# The unreserved URI characters (RFC 3986)
UNRESERVED_SET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" + "0123456789-._~"
)


def unquote_unreserved(uri):
    """Un-escape any percent-escape sequences in a URI that are unreserved
    characters. This leaves all reserved, illegal and non-ASCII bytes encoded.

    :rtype: str
    """
    parts = uri.split("%")
    for i in range(1, len(parts)):
        h = parts[i][0:2]
        if len(h) == 2 and h.isalnum():
            try:
                c = chr(int(h, 16))
            except ValueError:
                raise InvalidURL(f"Invalid percent-escape sequence: '{h}'")

            if c in UNRESERVED_SET:
                parts[i] = c + parts[i][2:]
            else:
                parts[i] = f"%{parts[i]}"
        else:
            parts[i] = f"%{parts[i]}"
    return "".join(parts)


def requote_uri(uri):
    """Re-quote the given URI.

    This function passes the given URI through an unquote/quote cycle to
    ensure that it is fully and consistently quoted.

    :rtype: str
    """
    safe_with_percent = "!#$%&'()*+,/:;=?@[]~"
    safe_without_percent = "!#$&'()*+,/:;=?@[]~"
    try:
        # Unquote only the unreserved characters
        # Then quote only illegal characters (do not quote reserved,
        # unreserved, or '%')
        return quote(unquote_unreserved(uri), safe=safe_with_percent)
    except InvalidURL:
        # We couldn't unquote the given URI, so let's try quoting it, but
        # there may be unquoted '%'s in the URI. We need to make sure they're
        # properly quoted so they do not cause issues elsewhere.
        return quote(uri, safe=safe_without_percent)


def address_in_network(ip, net):
    """This function allows you to check if an IP belongs to a network subnet

    Example: returns True if ip = 192.168.1.1 and net = 192.168.1.0/24
             returns False if ip = 192.168.1.1 and net = 192.168.100.0/24

    :rtype: bool
    """
    ipaddr = struct.unpack("=L", socket.inet_aton(ip))[0]
    netaddr, bits = net.split("/")
    netmask = struct.unpack("=L", socket.inet_aton(dotted_netmask(int(bits))))[0]
    network = struct.unpack("=L", socket.inet_aton(netaddr))[0] & netmask
    return (ipaddr & netmask) == (network & netmask)


def dotted_netmask(mask):
    """Converts mask from /xx format to xxx.xxx.xxx.xxx

    Example: if mask is 24 function returns 255.255.255.0

    :rtype: str
    """
    bits = 0xFFFFFFFF ^ (1 << 32 - mask) - 1
    return socket.inet_ntoa(struct.pack(">I", bits))


def is_ipv4_address(string_ip):
    """
    :rtype: bool
    """
    try:
        socket.inet_aton(string_ip)
    except OSError:
        return False
    return True


def is_valid_cidr(string_network):
    """
    Very simple check of the cidr format in no_proxy variable.

    :rtype: bool
    """
    if string_network.count("/") == 1:
        try:
            mask = int(string_network.split("/")[1])
        except ValueError:
            return False

        if mask < 1 or mask > 32:
            return False

        try:
            socket.inet_aton(string_network.split("/")[0])
        except OSError:
            return False
    else:
        return False
    return True


@contextlib.contextmanager
def set_environ(env_name, value):
    """Set the environment variable 'env_name' to 'value'

    Save previous value, yield, and then restore the previous value stored in
    the environment variable 'env_name'.

    If 'value' is None, do nothing"""
    value_changed = value is not None
    if value_changed:
        old_value = os.environ.get(env_name)
        os.environ[env_name] = value
    try:
        yield
    finally:
        if value_changed:
            if old_value is None:
                del os.environ[env_name]
            else:
                os.environ[env_name] = old_value


def should_bypass_proxies(url, no_proxy):
    """
    Returns whether we should bypass proxies or not.

    :rtype: bool
    """

    # Prioritize lowercase environment variables over uppercase
    # to keep a consistent behaviour with other http projects (curl, wget).
    def get_proxy(key):
        return os.environ.get(key) or os.environ.get(key.upper())

    # First check whether no_proxy is defined. If it is, check that the URL
    # we're getting isn't in the no_proxy list.
    no_proxy_arg = no_proxy
    if no_proxy is None:
        no_proxy = get_proxy("no_proxy")
    parsed = urlparse(url)

    if parsed.hostname is None:
        # URLs don't always have hostnames, e.g. file:/// urls.
        return True

    if no_proxy:
        # We need to check whether we match here. We need to see if we match
        # the end of the hostname, both with and without the port.
        no_proxy = (host for host in no_proxy.replace(" ", "").split(",") if host)

        if is_ipv4_address(parsed.hostname):
            for proxy_ip in no_proxy:
                if is_valid_cidr(proxy_ip):
                    if address_in_network(parsed.hostname, proxy_ip):
                        return True
                elif parsed.hostname == proxy_ip:
                    # If no_proxy ip was defined in plain IP notation instead of cidr notation &
                    # matches the IP of the index
                    return True
        else:
            host_with_port = parsed.hostname
            if parsed.port:
                host_with_port += f":{parsed.port}"

            for host in no_proxy:
                if parsed.hostname.endswith(host) or host_with_port.endswith(host):
                    # The URL does match something in no_proxy, so we don't want
                    # to apply the proxies on this URL.
                    return True

    with set_environ("no_proxy", no_proxy_arg):
        # parsed.hostname can be `None` in cases such as a file URI.
        try:
            bypass = proxy_bypass(parsed.hostname)
        except (TypeError, socket.gaierror):
            bypass = False

    if bypass:
        return True

    return False


def get_environ_proxies(url, no_proxy=None):
    """
    Return a dict of environment proxies.

    :rtype: dict
    """
    if should_bypass_proxies(url, no_proxy=no_proxy):
        return {}
    else:
        return getproxies()


def select_proxy(url, proxies):
    """Select a proxy for the url, if applicable.

    :param url: The url being for the request
    :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs
    """
    proxies = proxies or {}
    urlparts = urlparse(url)
    if urlparts.hostname is None:
        return proxies.get(urlparts.scheme, proxies.get("all"))

    proxy_keys = [
        urlparts.scheme + "://" + urlparts.hostname,
        urlparts.scheme,
        "all://" + urlparts.hostname,
        "all",
    ]
    proxy = None
    for proxy_key in proxy_keys:
        if proxy_key in proxies:
            proxy = proxies[proxy_key]
            break

    return proxy


def resolve_proxies(request, proxies, trust_env=True):
    """This method takes proxy information from a request and configuration
    input to resolve a mapping of target proxies. This will consider settings
    such as NO_PROXY to strip proxy configurations.

    :param request: Request or PreparedRequest
    :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs
    :param trust_env: Boolean declaring whether to trust environment configs

    :rtype: dict
    """
    proxies = proxies if proxies is not None else {}
    url = request.url
    scheme = urlparse(url).scheme
    no_proxy = proxies.get("no_proxy")
    new_proxies = proxies.copy()

    if trust_env and not should_bypass_proxies(url, no_proxy=no_proxy):
        environ_proxies = get_environ_proxies(url, no_proxy=no_proxy)

        proxy = environ_proxies.get(scheme, environ_proxies.get("all"))

        if proxy:
            new_proxies.setdefault(scheme, proxy)
    return new_proxies


def default_user_agent(name="python-requests"):
    """
    Return a string representing the default user agent.

    :rtype: str
    """
    return f"{name}/{__version__}"


def default_headers():
    """
    :rtype: requests.structures.CaseInsensitiveDict
    """
    return CaseInsensitiveDict(
        {
            "User-Agent": default_user_agent(),
            "Accept-Encoding": DEFAULT_ACCEPT_ENCODING,
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
    )


def parse_header_links(value):
    """Return a list of parsed link headers proxies.

    i.e. Link: <http:/.../front.jpeg>; rel=front; type="image/jpeg",<http://.../back.jpeg>; rel=back;type="image/jpeg"

    :rtype: list
    """

    links = []

    replace_chars = " '\""

    value = value.strip(replace_chars)
    if not value:
        return links

    for val in re.split(", *<", value):
        try:
            url, params = val.split(";", 1)
        except ValueError:
            url, params = val, ""

        link = {"url": url.strip("<> '\"")}

        for param in params.split(";"):
            try:
                key, value = param.split("=")
            except ValueError:
                break

            link[key.strip(replace_chars)] = value.strip(replace_chars)

        links.append(link)

    return links


# Null bytes; no need to recreate these on each call to guess_json_utf
_null = "\x00".encode("ascii")  # encoding to ASCII for Python 3
_null2 = _null * 2
_null3 = _null * 3


def guess_json_utf(data):
    """
    :rtype: str
    """
    # JSON always starts with two ASCII characters, so detection is as
    # easy as counting the nulls and from their location and count
    # determine the encoding. Also detect a BOM, if present.
    sample = data[:4]
    if sample in (codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE):
        return "utf-32"  # BOM included
    if sample[:3] == codecs.BOM_UTF8:
        return "utf-8-sig"  # BOM included, MS style (discouraged)
    if sample[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return "utf-16"  # BOM included
    nullcount = sample.count(_null)
    if nullcount == 0:
        return "utf-8"
    if nullcount == 2:
        if sample[::2] == _null2:  # 1st and 3rd are null
            return "utf-16-be"
        if sample[1::2] == _null2:  # 2nd and 4th are null
            return "utf-16-le"
        # Did not detect 2 valid UTF-16 ascii-range characters
    if nullcount == 3:
        if sample[:3] == _null3:
            return "utf-32-be"
        if sample[1:] == _null3:
            return "utf-32-le"
        # Did not detect a valid UTF-32 ascii-range character
    return None


def prepend_scheme_if_needed(url, new_scheme):
    """Given a URL that may or may not have a scheme, prepend the given scheme.
    Does not replace a present scheme with the one provided as an argument.

    :rtype: str
    """
    parsed = parse_url(url)
    scheme, auth, host, port, path, query, fragment = parsed

    # A defect in urlparse determines that there isn't a netloc present in some
    # urls. We previously assumed parsing was overly cautious, and swapped the
    # netloc and path. Due to a lack of tests on the original defect, this is
    # maintained with parse_url for backwards compatibility.
    netloc = parsed.netloc
    if not netloc:
        netloc, path = path, netloc

    if auth:
        # parse_url doesn't provide the netloc with auth
        # so we'll add it ourselves.
        netloc = "@".join([auth, netloc])
    if scheme is None:
        scheme = new_scheme
    if path is None:
        path = ""

    return urlunparse((scheme, netloc, path, "", query, fragment))


def get_auth_from_url(url):
    """Given a url with authentication components, extract them into a tuple of
    username,password.

    :rtype: (str,str)
    """
    parsed = urlparse(url)

    try:
        auth = (unquote(parsed.username), unquote(parsed.password))
    except (AttributeError, TypeError):
        auth = ("", "")

    return auth


def check_header_validity(header):
    """Verifies that header parts don't contain leading whitespace
    reserved characters, or return characters.

    :param header: tuple, in the format (name, value).
    """
    name, value = header
    _validate_header_part(header, name, 0)
    _validate_header_part(header, value, 1)


def _validate_header_part(header, header_part, header_validator_index):
    if isinstance(header_part, str):
        validator = _HEADER_VALIDATORS_STR[header_validator_index]
    elif isinstance(header_part, bytes):
        validator = _HEADER_VALIDATORS_BYTE[header_validator_index]
    else:
        raise InvalidHeader(
            f"Header part ({header_part!r}) from {header} "
            f"must be of type str or bytes, not {type(header_part)}"
        )

    if not validator.match(header_part):
        header_kind = "name" if header_validator_index == 0 else "value"
        raise InvalidHeader(
            f"Invalid leading whitespace, reserved character(s), or return "
            f"character(s) in header {header_kind}: {header_part!r}"
        )


def urldefragauth(url):
    """
    Given a url remove the fragment and the authentication part.

    :rtype: str
    """
    scheme, netloc, path, params, query, fragment = urlparse(url)

    # see func:`prepend_scheme_if_needed`
    if not netloc:
        netloc, path = path, netloc

    netloc = netloc.rsplit("@", 1)[-1]

    return urlunparse((scheme, netloc, path, params, query, ""))


def rewind_body(prepared_request):
    """Move file pointer back to its recorded starting position
    so it can be read again on redirect.
    """
    body_seek = getattr(prepared_request.body, "seek", None)
    if body_seek is not None and isinstance(
        prepared_request._body_position, integer_types
    ):
        try:
            body_seek(prepared_request._body_position)
        except OSError:
            raise UnrewindableBodyError(
                "An error occurred when rewinding request body for redirect."
            )
    else:
        raise UnrewindableBodyError("Unable to rewind request body for redirect.")


# === src/requests/exceptions.py ===
"""
requests.exceptions
~~~~~~~~~~~~~~~~~~~

This module contains the set of Requests' exceptions.
"""
from urllib3.exceptions import HTTPError as BaseHTTPError

from .compat import JSONDecodeError as CompatJSONDecodeError


class RequestException(IOError):
    """There was an ambiguous exception that occurred while handling your
    request.
    """

    def __init__(self, *args, **kwargs):
        """Initialize RequestException with `request` and `response` objects."""
        response = kwargs.pop("response", None)
        self.response = response
        self.request = kwargs.pop("request", None)
        if response is not None and not self.request and hasattr(response, "request"):
            self.request = self.response.request
        super().__init__(*args, **kwargs)


class InvalidJSONError(RequestException):
    """A JSON error occurred."""


class JSONDecodeError(InvalidJSONError, CompatJSONDecodeError):
    """Couldn't decode the text into json"""

    def __init__(self, *args, **kwargs):
        """
        Construct the JSONDecodeError instance first with all
        args. Then use it's args to construct the IOError so that
        the json specific args aren't used as IOError specific args
        and the error message from JSONDecodeError is preserved.
        """
        CompatJSONDecodeError.__init__(self, *args)
        InvalidJSONError.__init__(self, *self.args, **kwargs)

    def __reduce__(self):
        """
        The __reduce__ method called when pickling the object must
        be the one from the JSONDecodeError (be it json/simplejson)
        as it expects all the arguments for instantiation, not just
        one like the IOError, and the MRO would by default call the
        __reduce__ method from the IOError due to the inheritance order.
        """
        return CompatJSONDecodeError.__reduce__(self)


class HTTPError(RequestException):
    """An HTTP error occurred."""


class ConnectionError(RequestException):
    """A Connection error occurred."""


class ProxyError(ConnectionError):
    """A proxy error occurred."""


class SSLError(ConnectionError):
    """An SSL error occurred."""


class Timeout(RequestException):
    """The request timed out.

    Catching this error will catch both
    :exc:`~requests.exceptions.ConnectTimeout` and
    :exc:`~requests.exceptions.ReadTimeout` errors.
    """


class ConnectTimeout(ConnectionError, Timeout):
    """The request timed out while trying to connect to the remote server.

    Requests that produced this error are safe to retry.
    """


class ReadTimeout(Timeout):
    """The server did not send any data in the allotted amount of time."""


class URLRequired(RequestException):
    """A valid URL is required to make a request."""


class TooManyRedirects(RequestException):
    """Too many redirects."""


class MissingSchema(RequestException, ValueError):
    """The URL scheme (e.g. http or https) is missing."""


class InvalidSchema(RequestException, ValueError):
    """The URL scheme provided is either invalid or unsupported."""


class InvalidURL(RequestException, ValueError):
    """The URL provided was somehow invalid."""


class InvalidHeader(RequestException, ValueError):
    """The header value provided was somehow invalid."""


class InvalidProxyURL(InvalidURL):
    """The proxy URL provided is invalid."""


class ChunkedEncodingError(RequestException):
    """The server declared chunked encoding but sent an invalid chunk."""


class ContentDecodingError(RequestException, BaseHTTPError):
    """Failed to decode response content."""


class StreamConsumedError(RequestException, TypeError):
    """The content for this response was already consumed."""


class RetryError(RequestException):
    """Custom retries logic failed"""


class UnrewindableBodyError(RequestException):
    """Requests encountered an error when trying to rewind a body."""


# Warnings


class RequestsWarning(Warning):
    """Base warning for Requests."""


class FileModeWarning(RequestsWarning, DeprecationWarning):
    """A file was opened in text mode, but Requests determined its binary length."""


class RequestsDependencyWarning(RequestsWarning):
    """An imported dependency doesn't match the expected version range."""


# === src/requests/structures.py ===
"""
requests.structures
~~~~~~~~~~~~~~~~~~~

Data structures that power Requests.
"""

from collections import OrderedDict

from .compat import Mapping, MutableMapping


class CaseInsensitiveDict(MutableMapping):
    """A case-insensitive ``dict``-like object.

    Implements all methods and operations of
    ``MutableMapping`` as well as dict's ``copy``. Also
    provides ``lower_items``.

    All keys are expected to be strings. The structure remembers the
    case of the last key to be set, and ``iter(instance)``,
    ``keys()``, ``items()``, ``iterkeys()``, and ``iteritems()``
    will contain case-sensitive keys. However, querying and contains
    testing is case insensitive::

        cid = CaseInsensitiveDict()
        cid['Accept'] = 'application/json'
        cid['aCCEPT'] == 'application/json'  # True
        list(cid) == ['Accept']  # True

    For example, ``headers['content-encoding']`` will return the
    value of a ``'Content-Encoding'`` response header, regardless
    of how the header name was originally stored.

    If the constructor, ``.update``, or equality comparison
    operations are given keys that have equal ``.lower()``s, the
    behavior is undefined.
    """

    def __init__(self, data=None, **kwargs):
        self._store = OrderedDict()
        if data is None:
            data = {}
        self.update(data, **kwargs)

    def __setitem__(self, key, value):
        # Use the lowercased key for lookups, but store the actual
        # key alongside the value.
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key):
        return self._store[key.lower()][1]

    def __delitem__(self, key):
        del self._store[key.lower()]

    def __iter__(self):
        return (casedkey for casedkey, mappedvalue in self._store.values())

    def __len__(self):
        return len(self._store)

    def lower_items(self):
        """Like iteritems(), but with all lowercase keys."""
        return ((lowerkey, keyval[1]) for (lowerkey, keyval) in self._store.items())

    def __eq__(self, other):
        if isinstance(other, Mapping):
            other = CaseInsensitiveDict(other)
        else:
            return NotImplemented
        # Compare insensitively
        return dict(self.lower_items()) == dict(other.lower_items())

    # Copy is required
    def copy(self):
        return CaseInsensitiveDict(self._store.values())

    def __repr__(self):
        return str(dict(self.items()))


class LookupDict(dict):
    """Dictionary lookup object."""

    def __init__(self, name=None):
        self.name = name
        super().__init__()

    def __repr__(self):
        return f"<lookup '{self.name}'>"

    def __getitem__(self, key):
        # We allow fall-through here, so values default to None

        return self.__dict__.get(key, None)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


# === src/requests/help.py ===
"""Module containing bug report helper(s)."""

import json
import platform
import ssl
import sys

import idna
import urllib3

from . import __version__ as requests_version

try:
    import charset_normalizer
except ImportError:
    charset_normalizer = None

try:
    import chardet
except ImportError:
    chardet = None

try:
    from urllib3.contrib import pyopenssl
except ImportError:
    pyopenssl = None
    OpenSSL = None
    cryptography = None
else:
    import cryptography
    import OpenSSL


def _implementation():
    """Return a dict with the Python implementation and version.

    Provide both the name and the version of the Python implementation
    currently running. For example, on CPython 3.10.3 it will return
    {'name': 'CPython', 'version': '3.10.3'}.

    This function works best on CPython and PyPy: in particular, it probably
    doesn't work for Jython or IronPython. Future investigation should be done
    to work out the correct shape of the code for those platforms.
    """
    implementation = platform.python_implementation()

    if implementation == "CPython":
        implementation_version = platform.python_version()
    elif implementation == "PyPy":
        implementation_version = "{}.{}.{}".format(
            sys.pypy_version_info.major,
            sys.pypy_version_info.minor,
            sys.pypy_version_info.micro,
        )
        if sys.pypy_version_info.releaselevel != "final":
            implementation_version = "".join(
                [implementation_version, sys.pypy_version_info.releaselevel]
            )
    elif implementation == "Jython":
        implementation_version = platform.python_version()  # Complete Guess
    elif implementation == "IronPython":
        implementation_version = platform.python_version()  # Complete Guess
    else:
        implementation_version = "Unknown"

    return {"name": implementation, "version": implementation_version}


def info():
    """Generate information for a bug report."""
    try:
        platform_info = {
            "system": platform.system(),
            "release": platform.release(),
        }
    except OSError:
        platform_info = {
            "system": "Unknown",
            "release": "Unknown",
        }

    implementation_info = _implementation()
    urllib3_info = {"version": urllib3.__version__}
    charset_normalizer_info = {"version": None}
    chardet_info = {"version": None}
    if charset_normalizer:
        charset_normalizer_info = {"version": charset_normalizer.__version__}
    if chardet:
        chardet_info = {"version": chardet.__version__}

    pyopenssl_info = {
        "version": None,
        "openssl_version": "",
    }
    if OpenSSL:
        pyopenssl_info = {
            "version": OpenSSL.__version__,
            "openssl_version": f"{OpenSSL.SSL.OPENSSL_VERSION_NUMBER:x}",
        }
    cryptography_info = {
        "version": getattr(cryptography, "__version__", ""),
    }
    idna_info = {
        "version": getattr(idna, "__version__", ""),
    }

    system_ssl = ssl.OPENSSL_VERSION_NUMBER
    system_ssl_info = {"version": f"{system_ssl:x}" if system_ssl is not None else ""}

    return {
        "platform": platform_info,
        "implementation": implementation_info,
        "system_ssl": system_ssl_info,
        "using_pyopenssl": pyopenssl is not None,
        "using_charset_normalizer": chardet is None,
        "pyOpenSSL": pyopenssl_info,
        "urllib3": urllib3_info,
        "chardet": chardet_info,
        "charset_normalizer": charset_normalizer_info,
        "cryptography": cryptography_info,
        "idna": idna_info,
        "requests": {
            "version": requests_version,
        },
    }


def main():
    """Pretty-print the bug information as JSON."""
    print(json.dumps(info(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()


# === src/requests/adapters.py ===
"""
requests.adapters
~~~~~~~~~~~~~~~~~

This module contains the transport adapters that Requests uses to define
and maintain connections.
"""

import os.path
import socket  # noqa: F401
import typing
import warnings

from urllib3.exceptions import ClosedPoolError, ConnectTimeoutError
from urllib3.exceptions import HTTPError as _HTTPError
from urllib3.exceptions import InvalidHeader as _InvalidHeader
from urllib3.exceptions import (
    LocationValueError,
    MaxRetryError,
    NewConnectionError,
    ProtocolError,
)
from urllib3.exceptions import ProxyError as _ProxyError
from urllib3.exceptions import ReadTimeoutError, ResponseError
from urllib3.exceptions import SSLError as _SSLError
from urllib3.poolmanager import PoolManager, proxy_from_url
from urllib3.util import Timeout as TimeoutSauce
from urllib3.util import parse_url
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import create_urllib3_context

from .auth import _basic_auth_str
from .compat import basestring, urlparse
from .cookies import extract_cookies_to_jar
from .exceptions import (
    ConnectionError,
    ConnectTimeout,
    InvalidHeader,
    InvalidProxyURL,
    InvalidSchema,
    InvalidURL,
    ProxyError,
    ReadTimeout,
    RetryError,
    SSLError,
)
from .models import Response
from .structures import CaseInsensitiveDict
from .utils import (
    DEFAULT_CA_BUNDLE_PATH,
    extract_zipped_paths,
    get_auth_from_url,
    get_encoding_from_headers,
    prepend_scheme_if_needed,
    select_proxy,
    urldefragauth,
)

try:
    from urllib3.contrib.socks import SOCKSProxyManager
except ImportError:

    def SOCKSProxyManager(*args, **kwargs):
        raise InvalidSchema("Missing dependencies for SOCKS support.")


if typing.TYPE_CHECKING:
    from .models import PreparedRequest


DEFAULT_POOLBLOCK = False
DEFAULT_POOLSIZE = 10
DEFAULT_RETRIES = 0
DEFAULT_POOL_TIMEOUT = None


try:
    import ssl  # noqa: F401

    _preloaded_ssl_context = create_urllib3_context()
    _preloaded_ssl_context.load_verify_locations(
        extract_zipped_paths(DEFAULT_CA_BUNDLE_PATH)
    )
except ImportError:
    # Bypass default SSLContext creation when Python
    # interpreter isn't built with the ssl module.
    _preloaded_ssl_context = None


def _urllib3_request_context(
    request: "PreparedRequest",
    verify: "bool | str | None",
    client_cert: "typing.Tuple[str, str] | str | None",
    poolmanager: "PoolManager",
) -> "(typing.Dict[str, typing.Any], typing.Dict[str, typing.Any])":
    host_params = {}
    pool_kwargs = {}
    parsed_request_url = urlparse(request.url)
    scheme = parsed_request_url.scheme.lower()
    port = parsed_request_url.port

    # Determine if we have and should use our default SSLContext
    # to optimize performance on standard requests.
    poolmanager_kwargs = getattr(poolmanager, "connection_pool_kw", {})
    has_poolmanager_ssl_context = poolmanager_kwargs.get("ssl_context")
    should_use_default_ssl_context = (
        _preloaded_ssl_context is not None and not has_poolmanager_ssl_context
    )

    cert_reqs = "CERT_REQUIRED"
    if verify is False:
        cert_reqs = "CERT_NONE"
    elif verify is True and should_use_default_ssl_context:
        pool_kwargs["ssl_context"] = _preloaded_ssl_context
    elif isinstance(verify, str):
        if not os.path.isdir(verify):
            pool_kwargs["ca_certs"] = verify
        else:
            pool_kwargs["ca_cert_dir"] = verify
    pool_kwargs["cert_reqs"] = cert_reqs
    if client_cert is not None:
        if isinstance(client_cert, tuple) and len(client_cert) == 2:
            pool_kwargs["cert_file"] = client_cert[0]
            pool_kwargs["key_file"] = client_cert[1]
        else:
            # According to our docs, we allow users to specify just the client
            # cert path
            pool_kwargs["cert_file"] = client_cert
    host_params = {
        "scheme": scheme,
        "host": parsed_request_url.hostname,
        "port": port,
    }
    return host_params, pool_kwargs


class BaseAdapter:
    """The Base Transport Adapter"""

    def __init__(self):
        super().__init__()

    def send(
        self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None
    ):
        """Sends PreparedRequest object. Returns Response object.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        """
        raise NotImplementedError

    def close(self):
        """Cleans up adapter specific items."""
        raise NotImplementedError


class HTTPAdapter(BaseAdapter):
    """The built-in HTTP Adapter for urllib3.

    Provides a general-case interface for Requests sessions to contact HTTP and
    HTTPS urls by implementing the Transport Adapter interface. This class will
    usually be created by the :class:`Session <Session>` class under the
    covers.

    :param pool_connections: The number of urllib3 connection pools to cache.
    :param pool_maxsize: The maximum number of connections to save in the pool.
    :param max_retries: The maximum number of retries each connection
        should attempt. Note, this applies only to failed DNS lookups, socket
        connections and connection timeouts, never to requests where data has
        made it to the server. By default, Requests does not retry failed
        connections. If you need granular control over the conditions under
        which we retry a request, import urllib3's ``Retry`` class and pass
        that instead.
    :param pool_block: Whether the connection pool should block for connections.

    Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> a = requests.adapters.HTTPAdapter(max_retries=3)
      >>> s.mount('http://', a)
    """

    __attrs__ = [
        "max_retries",
        "config",
        "_pool_connections",
        "_pool_maxsize",
        "_pool_block",
    ]

    def __init__(
        self,
        pool_connections=DEFAULT_POOLSIZE,
        pool_maxsize=DEFAULT_POOLSIZE,
        max_retries=DEFAULT_RETRIES,
        pool_block=DEFAULT_POOLBLOCK,
    ):
        if max_retries == DEFAULT_RETRIES:
            self.max_retries = Retry(0, read=False)
        else:
            self.max_retries = Retry.from_int(max_retries)
        self.config = {}
        self.proxy_manager = {}

        super().__init__()

        self._pool_connections = pool_connections
        self._pool_maxsize = pool_maxsize
        self._pool_block = pool_block

        self.init_poolmanager(pool_connections, pool_maxsize, block=pool_block)

    def __getstate__(self):
        return {attr: getattr(self, attr, None) for attr in self.__attrs__}

    def __setstate__(self, state):
        # Can't handle by adding 'proxy_manager' to self.__attrs__ because
        # self.poolmanager uses a lambda function, which isn't pickleable.
        self.proxy_manager = {}
        self.config = {}

        for attr, value in state.items():
            setattr(self, attr, value)

        self.init_poolmanager(
            self._pool_connections, self._pool_maxsize, block=self._pool_block
        )

    def init_poolmanager(
        self, connections, maxsize, block=DEFAULT_POOLBLOCK, **pool_kwargs
    ):
        """Initializes a urllib3 PoolManager.

        This method should not be called from user code, and is only
        exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param connections: The number of urllib3 connection pools to cache.
        :param maxsize: The maximum number of connections to save in the pool.
        :param block: Block when no free connections are available.
        :param pool_kwargs: Extra keyword arguments used to initialize the Pool Manager.
        """
        # save these values for pickling
        self._pool_connections = connections
        self._pool_maxsize = maxsize
        self._pool_block = block

        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        """Return urllib3 ProxyManager for the given proxy.

        This method should not be called from user code, and is only
        exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param proxy: The proxy to return a urllib3 ProxyManager for.
        :param proxy_kwargs: Extra keyword arguments used to configure the Proxy Manager.
        :returns: ProxyManager
        :rtype: urllib3.ProxyManager
        """
        if proxy in self.proxy_manager:
            manager = self.proxy_manager[proxy]
        elif proxy.lower().startswith("socks"):
            username, password = get_auth_from_url(proxy)
            manager = self.proxy_manager[proxy] = SOCKSProxyManager(
                proxy,
                username=username,
                password=password,
                num_pools=self._pool_connections,
                maxsize=self._pool_maxsize,
                block=self._pool_block,
                **proxy_kwargs,
            )
        else:
            proxy_headers = self.proxy_headers(proxy)
            manager = self.proxy_manager[proxy] = proxy_from_url(
                proxy,
                proxy_headers=proxy_headers,
                num_pools=self._pool_connections,
                maxsize=self._pool_maxsize,
                block=self._pool_block,
                **proxy_kwargs,
            )

        return manager

    def cert_verify(self, conn, url, verify, cert):
        """Verify a SSL certificate. This method should not be called from user
        code, and is only exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param conn: The urllib3 connection object associated with the cert.
        :param url: The requested URL.
        :param verify: Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use
        :param cert: The SSL certificate to verify.
        """
        if url.lower().startswith("https") and verify:
            conn.cert_reqs = "CERT_REQUIRED"

            # Only load the CA certificates if 'verify' is a string indicating the CA bundle to use.
            # Otherwise, if verify is a boolean, we don't load anything since
            # the connection will be using a context with the default certificates already loaded,
            # and this avoids a call to the slow load_verify_locations()
            if verify is not True:
                # `verify` must be a str with a path then
                cert_loc = verify

                if not os.path.exists(cert_loc):
                    raise OSError(
                        f"Could not find a suitable TLS CA certificate bundle, "
                        f"invalid path: {cert_loc}"
                    )

                if not os.path.isdir(cert_loc):
                    conn.ca_certs = cert_loc
                else:
                    conn.ca_cert_dir = cert_loc
        else:
            conn.cert_reqs = "CERT_NONE"
            conn.ca_certs = None
            conn.ca_cert_dir = None

        if cert:
            if not isinstance(cert, basestring):
                conn.cert_file = cert[0]
                conn.key_file = cert[1]
            else:
                conn.cert_file = cert
                conn.key_file = None
            if conn.cert_file and not os.path.exists(conn.cert_file):
                raise OSError(
                    f"Could not find the TLS certificate file, "
                    f"invalid path: {conn.cert_file}"
                )
            if conn.key_file and not os.path.exists(conn.key_file):
                raise OSError(
                    f"Could not find the TLS key file, invalid path: {conn.key_file}"
                )

    def build_response(self, req, resp):
        """Builds a :class:`Response <requests.Response>` object from a urllib3
        response. This should not be called from user code, and is only exposed
        for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`

        :param req: The :class:`PreparedRequest <PreparedRequest>` used to generate the response.
        :param resp: The urllib3 response object.
        :rtype: requests.Response
        """
        response = Response()

        # Fallback to None if there's no status_code, for whatever reason.
        response.status_code = getattr(resp, "status", None)

        # Make headers case-insensitive.
        response.headers = CaseInsensitiveDict(getattr(resp, "headers", {}))

        # Set encoding.
        response.encoding = get_encoding_from_headers(response.headers)
        response.raw = resp
        response.reason = response.raw.reason

        if isinstance(req.url, bytes):
            response.url = req.url.decode("utf-8")
        else:
            response.url = req.url

        # Add new cookies from the server.
        extract_cookies_to_jar(response.cookies, req, resp)

        # Give the Response some context.
        response.request = req
        response.connection = self

        return response

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        """Build the PoolKey attributes used by urllib3 to return a connection.

        This looks at the PreparedRequest, the user-specified verify value,
        and the value of the cert parameter to determine what PoolKey values
        to use to select a connection from a given urllib3 Connection Pool.

        The SSL related pool key arguments are not consistently set. As of
        this writing, use the following to determine what keys may be in that
        dictionary:

        * If ``verify`` is ``True``, ``"ssl_context"`` will be set and will be the
          default Requests SSL Context
        * If ``verify`` is ``False``, ``"ssl_context"`` will not be set but
          ``"cert_reqs"`` will be set
        * If ``verify`` is a string, (i.e., it is a user-specified trust bundle)
          ``"ca_certs"`` will be set if the string is not a directory recognized
          by :py:func:`os.path.isdir`, otherwise ``"ca_certs_dir"`` will be
          set.
        * If ``"cert"`` is specified, ``"cert_file"`` will always be set. If
          ``"cert"`` is a tuple with a second item, ``"key_file"`` will also
          be present

        To override these settings, one may subclass this class, call this
        method and use the above logic to change parameters as desired. For
        example, if one wishes to use a custom :py:class:`ssl.SSLContext` one
        must both set ``"ssl_context"`` and based on what else they require,
        alter the other keys to ensure the desired behaviour.

        :param request:
            The PreparedReqest being sent over the connection.
        :type request:
            :class:`~requests.models.PreparedRequest`
        :param verify:
            Either a boolean, in which case it controls whether
            we verify the server's TLS certificate, or a string, in which case it
            must be a path to a CA bundle to use.
        :param cert:
            (optional) Any user-provided SSL certificate for client
            authentication (a.k.a., mTLS). This may be a string (i.e., just
            the path to a file which holds both certificate and key) or a
            tuple of length 2 with the certificate file path and key file
            path.
        :returns:
            A tuple of two dictionaries. The first is the "host parameters"
            portion of the Pool Key including scheme, hostname, and port. The
            second is a dictionary of SSLContext related parameters.
        """
        return _urllib3_request_context(request, verify, cert, self.poolmanager)

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        """Returns a urllib3 connection for the given request and TLS settings.
        This should not be called from user code, and is only exposed for use
        when subclassing the :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request:
            The :class:`PreparedRequest <PreparedRequest>` object to be sent
            over the connection.
        :param verify:
            Either a boolean, in which case it controls whether we verify the
            server's TLS certificate, or a string, in which case it must be a
            path to a CA bundle to use.
        :param proxies:
            (optional) The proxies dictionary to apply to the request.
        :param cert:
            (optional) Any user-provided SSL certificate to be used for client
            authentication (a.k.a., mTLS).
        :rtype:
            urllib3.ConnectionPool
        """
        proxy = select_proxy(request.url, proxies)
        try:
            host_params, pool_kwargs = self.build_connection_pool_key_attributes(
                request,
                verify,
                cert,
            )
        except ValueError as e:
            raise InvalidURL(e, request=request)
        if proxy:
            proxy = prepend_scheme_if_needed(proxy, "http")
            proxy_url = parse_url(proxy)
            if not proxy_url.host:
                raise InvalidProxyURL(
                    "Please check proxy URL. It is malformed "
                    "and could be missing the host."
                )
            proxy_manager = self.proxy_manager_for(proxy)
            conn = proxy_manager.connection_from_host(
                **host_params, pool_kwargs=pool_kwargs
            )
        else:
            # Only scheme should be lower case
            conn = self.poolmanager.connection_from_host(
                **host_params, pool_kwargs=pool_kwargs
            )

        return conn

    def get_connection(self, url, proxies=None):
        """DEPRECATED: Users should move to `get_connection_with_tls_context`
        for all subclasses of HTTPAdapter using Requests>=2.32.2.

        Returns a urllib3 connection for the given URL. This should not be
        called from user code, and is only exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param url: The URL to connect to.
        :param proxies: (optional) A Requests-style dictionary of proxies used on this request.
        :rtype: urllib3.ConnectionPool
        """
        warnings.warn(
            (
                "`get_connection` has been deprecated in favor of "
                "`get_connection_with_tls_context`. Custom HTTPAdapter subclasses "
                "will need to migrate for Requests>=2.32.2. Please see "
                "https://github.com/psf/requests/pull/6710 for more details."
            ),
            DeprecationWarning,
        )
        proxy = select_proxy(url, proxies)

        if proxy:
            proxy = prepend_scheme_if_needed(proxy, "http")
            proxy_url = parse_url(proxy)
            if not proxy_url.host:
                raise InvalidProxyURL(
                    "Please check proxy URL. It is malformed "
                    "and could be missing the host."
                )
            proxy_manager = self.proxy_manager_for(proxy)
            conn = proxy_manager.connection_from_url(url)
        else:
            # Only scheme should be lower case
            parsed = urlparse(url)
            url = parsed.geturl()
            conn = self.poolmanager.connection_from_url(url)

        return conn

    def close(self):
        """Disposes of any internal state.

        Currently, this closes the PoolManager and any active ProxyManager,
        which closes any pooled connections.
        """
        self.poolmanager.clear()
        for proxy in self.proxy_manager.values():
            proxy.clear()

    def request_url(self, request, proxies):
        """Obtain the url to use when making the final request.

        If the message is being sent through a HTTP proxy, the full URL has to
        be used. Otherwise, we should only use the path portion of the URL.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs.
        :rtype: str
        """
        proxy = select_proxy(request.url, proxies)
        scheme = urlparse(request.url).scheme

        is_proxied_http_request = proxy and scheme != "https"
        using_socks_proxy = False
        if proxy:
            proxy_scheme = urlparse(proxy).scheme.lower()
            using_socks_proxy = proxy_scheme.startswith("socks")

        url = request.path_url
        if url.startswith("//"):  # Don't confuse urllib3
            url = f"/{url.lstrip('/')}"

        if is_proxied_http_request and not using_socks_proxy:
            url = urldefragauth(request.url)

        return url

    def add_headers(self, request, **kwargs):
        """Add any headers needed by the connection. As of v2.0 this does
        nothing by default, but is left for overriding by users that subclass
        the :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request: The :class:`PreparedRequest <PreparedRequest>` to add headers to.
        :param kwargs: The keyword arguments from the call to send().
        """
        pass

    def proxy_headers(self, proxy):
        """Returns a dictionary of the headers to add to any request sent
        through a proxy. This works with urllib3 magic to ensure that they are
        correctly sent to the proxy, rather than in a tunnelled request if
        CONNECT is being used.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param proxy: The url of the proxy being used for this request.
        :rtype: dict
        """
        headers = {}
        username, password = get_auth_from_url(proxy)

        if username:
            headers["Proxy-Authorization"] = _basic_auth_str(username, password)

        return headers

    def send(
        self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None
    ):
        """Sends PreparedRequest object. Returns Response object.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple or urllib3 Timeout object
        :param verify: (optional) Either a boolean, in which case it controls whether
            we verify the server's TLS certificate, or a string, in which case it
            must be a path to a CA bundle to use
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        :rtype: requests.Response
        """

        try:
            conn = self.get_connection_with_tls_context(
                request, verify, proxies=proxies, cert=cert
            )
        except LocationValueError as e:
            raise InvalidURL(e, request=request)

        self.cert_verify(conn, request.url, verify, cert)
        url = self.request_url(request, proxies)
        self.add_headers(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies,
        )

        chunked = not (request.body is None or "Content-Length" in request.headers)

        if isinstance(timeout, tuple):
            try:
                connect, read = timeout
                timeout = TimeoutSauce(connect=connect, read=read)
            except ValueError:
                raise ValueError(
                    f"Invalid timeout {timeout}. Pass a (connect, read) timeout tuple, "
                    f"or a single float to set both timeouts to the same value."
                )
        elif isinstance(timeout, TimeoutSauce):
            pass
        else:
            timeout = TimeoutSauce(connect=timeout, read=timeout)

        try:
            resp = conn.urlopen(
                method=request.method,
                url=url,
                body=request.body,
                headers=request.headers,
                redirect=False,
                assert_same_host=False,
                preload_content=False,
                decode_content=False,
                retries=self.max_retries,
                timeout=timeout,
                chunked=chunked,
            )

        except (ProtocolError, OSError) as err:
            raise ConnectionError(err, request=request)

        except MaxRetryError as e:
            if isinstance(e.reason, ConnectTimeoutError):
                # TODO: Remove this in 3.0.0: see #2811
                if not isinstance(e.reason, NewConnectionError):
                    raise ConnectTimeout(e, request=request)

            if isinstance(e.reason, ResponseError):
                raise RetryError(e, request=request)

            if isinstance(e.reason, _ProxyError):
                raise ProxyError(e, request=request)

            if isinstance(e.reason, _SSLError):
                # This branch is for urllib3 v1.22 and later.
                raise SSLError(e, request=request)

            raise ConnectionError(e, request=request)

        except ClosedPoolError as e:
            raise ConnectionError(e, request=request)

        except _ProxyError as e:
            raise ProxyError(e)

        except (_SSLError, _HTTPError) as e:
            if isinstance(e, _SSLError):
                # This branch is for urllib3 versions earlier than v1.22
                raise SSLError(e, request=request)
            elif isinstance(e, ReadTimeoutError):
                raise ReadTimeout(e, request=request)
            elif isinstance(e, _InvalidHeader):
                raise InvalidHeader(e, request=request)
            else:
                raise

        return self.build_response(request, resp)
