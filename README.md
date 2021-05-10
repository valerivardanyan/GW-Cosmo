# GW-Cosmo

Analysis pipeline accompanying Canas-Herrera, Contigiani, Vardanyan [arXiv:]. Please cite this paper when making use of the repository. 


Prerequisites
----------

* [COLOSSUS](https://bdiemer.bitbucket.io/colossus/)
* [Astropy](https://www.astropy.org/)
* [sklearn](https://scikit-learn.org/)
* [emcee](https://emcee.readthedocs.io/)
* [GetDist](https://getdist.readthedocs.io/)


Main files
----------

* /src/GWGC_cross_corr.py : The main source file. Calculates the theoretical spectra and covariances, generates mock data.  

* /Plotting/Example.ipynb : Example notebook.

* /MCMC_run/specifications.py : Survey specifications are given here. 
* /MCMC_run/initialize.py : Performs some initializations. 
* /MCMC_run/chain.py : Contains the likelihood and GP setup. 
* /MCMC_run/check-fit.ipynb : Monitors the chain status and makes preliminary plots. 


