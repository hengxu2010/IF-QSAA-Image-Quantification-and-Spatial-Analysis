# IF-QSAA: Image Quantification and Spatial Analysis

Python tools for image quantification and spatial analysis in IF-QSAA (immunofluorescence-integrated quiescent seed amplification assay). The toolkit analyzes amplified products and immunofluorescence markers measured on the same tissue section.

The package includes image loading, background correction, segmentation, spectral cross-talk correction, object-to-territory contact analysis, object pairing, boundary-distance profiles, and statistical procedures.

The implementation and detailed documentation are in [`if-qsaa-toolkit/`](if-qsaa-toolkit/).

## Quick start

Requires Python 3.9 or later. After cloning or downloading this repository, open a terminal in its root directory and run:

```sh
cd if-qsaa-toolkit
python -m pip install -e .
python examples/contact_demo.py
```

The example generates synthetic data and runs a contact analysis. Additional examples cover segmentation, spectral unmixing, the learned detector, object pairing, and boundary-distance profiles.

For your own data, supply the appropriate pixel size, spectral coupling coefficients, intensity settings, and experimental labels. See the [parameter guide](if-qsaa-toolkit/README.md#how-the-parameters-work).

## Documentation and code

- [Full documentation](if-qsaa-toolkit/README.md)
- [Python source](if-qsaa-toolkit/src/if_qsaa/)
- [Runnable examples](if-qsaa-toolkit/examples/)
- [Test suite](if-qsaa-toolkit/tests/)
- [Package dependencies](if-qsaa-toolkit/pyproject.toml)
- [Citation information](if-qsaa-toolkit/README.md#citation)

## Run the tests

From the `if-qsaa-toolkit` directory:

```sh
python -m pip install -e ".[test]"
python -m pytest -q
```

## License

The toolkit is available under the [MIT License](if-qsaa-toolkit/LICENSE).
