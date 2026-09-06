# Third-party sources and licensing

The MIT license applies to original OSMO source code and documentation. It does not grant rights to external datasets or change dependency licenses. No third-party training dataset, subject-level table, trained model weight, upstream source-code bundle or PK-Sim project file is distributed here.

Research references:

- [Open Systems Pharmacology](https://www.open-systems-pharmacology.org/) and [PK-Sim](https://github.com/Open-Systems-Pharmacology/PK-Sim): public PBPK tools and documentation. Upstream software has its own license; observed measurements may have separate publication rights. The OSMO legacy-array reader is original format-reading code, not a bundled PK-Sim implementation.
- [EPA HTTK](https://github.com/USEPA/httk): context for CvTdb concentration data. Review the versioned data source and source publications separately.
- [ChEMBL](https://www.ebi.ac.uk/chembl/): earlier ADME observations carried CC-BY-SA-3.0 attribution in the local research manifests. They are not relicensed or redistributed here.
- [PKSmart](https://github.com/srijitseal/PKSmart), Seal et al., [2024.02.02.578658](https://doi.org/10.1101/2024.02.02.578658): external repository carrying Apache-2.0 terms. OSMO's auxiliary experiments used tabulated properties; no PKSmart source code or pretrained weights are included. Original study provenance still matters.
- Dryad data: [empagliflozin](https://doi.org/10.5061/dryad.brv15dv8j), [7,8-DHF](https://doi.org/10.5061/dryad.fbg79cp5q), [dantrolene](https://doi.org/10.5061/dryad.jdfn2z37k), [morphine](https://doi.org/10.5061/dryad.m8v146p). These informed local data-adapter work; their raw measurements are not part of this release.

Python dependencies are installed separately through their distributions and retain their own copyright notices and licenses. References do not imply author affiliation, endorsement or reproduction of published results.
