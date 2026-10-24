# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- build: restrict supported Python versions to 3.12 and 3.13; 3.14 was never installable with the pinned dependencies
- fix: cap torch below 2.11 so its CUDA 13 NCCL library does not clash with xgboost's CUDA 12 NCCL on Linux
- refactor: rename sequence to workflow
- changed name of output directory

## [0.1.0] -

### Added

- Added `num_features`, `cat_nominal_features`, and `cat_ordinal_features` properties to OctoData class to retrieve feature columns by type
