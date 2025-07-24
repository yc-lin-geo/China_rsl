# Modern sea-level rise breaks 4000-year stability in southeastern China

Yucheng Lin, Robert E. Kopp, Haixian Xiong, Fiona D. Hibbert, Zhuo Zheng, Fengling Yu, Praveen Kumar, Sönke Dangendorf, Hailin Yi and Yaze Zhang

yc.lin@csiro.au

## Paper Abstract

Quantifying physical mechanisms driving sea-level change—including global mean sea level (GMSL) and regional-to-local components (i.e., sea level budget)—is essential for reliable future projections and effective coastal management. While previous research has attempted to resolve China’s sea-level budget from the 1950s, these studies capture short timescales and lack the long-term context necessary to fully assess modern sea-level rise in southeastern China—one of the world’s most densely populated regions with immense socioeconomic importance. Here we show that GMSL followed three distinct stages from 11,700 years before present (BP) to the modern day: (1) rapid early Holocene rise driven by the deglacial melt of land ice, (2) 4000 years of stability from ~4200 BP to mid-19th century when regional processes dominated sea-level change, and (3) accelerating rise from the mid-19th century. Our results arise from spatiotemporal hierarchical modeling of geological sea-level proxies and tide gauge data to produce site-specific sea-level budget estimates with uncertainty quantification. It is extremely likely (P$\ge$0.95) that the GMSL rise rate since 1900 (1.51 +/1 0.16 mm/yr, 1sigma) has exceeded any century over at least the past four millennia. Moreover, our analysis indicates that at least 94\% of rapid modern urban subsidence is attributable to anthropogenic activities, with localized subsidence rates often exceeding GMSL rise. Such concurrent acceleration of global sea-level rise and rapid localized subsidence has not been observed in our Holocene geological record.

## Code environment

The spatiotemporal hierarchical model used in this study is built upon  [PaleoSTeHM](https://github.com/radical-collaboration/PaleoSTeHM) framework (last access, 1/May/2025). To install:

Create and activate a Python virtual environment, and install PaleoSTeHM's Python 
dependencies in it. Using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment):

```
conda env create -f environment.yml
conda activate ve3PaleoSTeHM
```



## Notebooks

1. - **[Load and visulize sea-level data](notebooks/RSL_data.ipynb)** - A notebook used to load and visulize sea-level index points database compiled in this study.

2. - **[GMSL trend and rate reconstructions](notebooks/GMSL_trend_rate.ipynb)** - A notebook showing global mean sea-level reconstruction in this study, which is compared with ohter different previous models. A GMSL rate plot is given over the Holocene.

3. - **[VLM analysis](notebooks/VLM_analysis.ipynb)** - A notebook showing geological (or natural) vertical land motion reconstruction by this study, which is compared with modern InSAR and GNSS data. 

4. - **[Holocene sea-level budget reconstruction results](notebooks/Budget_recon.ipynb)** - A notebook showing renstructed sea-level budget results based on hierarchical modeling presented in this study.

5. - **[Analytical model for sterodynamic sea level at Hangzhou](notebooks/Analytical_SDSL.ipynb)** - A notebook that use an analytical model to investigate river discharge impact on sterodynamic sea level at Hangzhou. 

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details

## Acknowledgements

We thank Chris Piecuch for his help regarding river discharge impact on sea-level, Michael Stein for the helpful discussions about spatiotemporal statistics. Y.L. and R.E.K. were supported by the U.S. National Science Foundation under awards 2002437 and 2148265, the latter as part of the Megalopolitan Coastal Transformation Hub (MACH). This is MACH contribution number 86. R.E.K. was also supported by the U.S. National Aeronautics and Space Administration, as part of the NASA Sea-Level Change Team under JPL task 105393.509496.02.08.13.31. The authors acknowledge PALSEA, a working group of the International Union for Quaternary Sciences (INQUA) and Past Global Changes (PAGES), which received support from the Swiss Academy of Sciences and the Chinese Academy of Sciences.
