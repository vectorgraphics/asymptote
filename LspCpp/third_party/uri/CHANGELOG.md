# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2018-11-24
### Fixed
- Query percent encoding

## [1.0.4] - 2018-10-21
### Fixed
- Bug in path normalization

## [1.0.3] - 2018-10-18
### Added
- AppVeyor for Visual Studio 2015

### Fixed
- Bug in percent encoding non-ASCII characters
- Percent encoding query part

## [1.0.2] - 2018-10-13
### Fixed
- Bug in `string_view` implementation
- Incorrect port copy constructor implementation

## [1.0.1] - 2018-08-11
### Changed
- Build defaults to C++11

### Fixed
- Fix to `network::uri_builder` to allow URIs that have a scheme and absolute path
- Other minor bug fixes and optimizations

## [1.0.0] - 2018-05-27
### Added
- A class, `network::uri` that models a URI, including URI parsing on construction
  according to [RFC 3986](https://tools.ietf.org/html/rfc3986)
- A class, `network::uri_builder` that allows a user to construct valid URIs
- Member functions to allow URI normalization, resolution, and comparison
- Support for URI percent encoding
