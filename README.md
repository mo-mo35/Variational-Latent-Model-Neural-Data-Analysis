# LVM-Neural-Data-Analysis

Since we gathered the IBL data using the ONE api there is no need to download data locally before running the code. However, Docker is required to launch the environment.

Once you have docker installed, find a directory and clone the repository using
'git clone https://github.com/Charlie-279/LVM-Neural-Data-Analysis.git'

Repository cloned, in your terminal find the main directory and use the following command:
"docker build -t 'name_of_proj_env' ."

then,

"docker run -it --rm 'name_of_proj_env'

To run the project simply run both python files in sequence, run_vlgp.py and run_pcca.py. 
e.g. 'python run_vlgp.py'

If prompted for password for Alyx, use "international"

Parameters are held in the forked vlgp repo, in preprocess.py, originally made by catniplab, see references and requirements.txt.


config.json holds parameters for brain regions to search for and time interval sizes.



References:


Alessio P. Buccino, et al. (2022). Progress in Biomed. Eng., 4(022005). https://iopscience.iop.org/article/10.1088/2516-1091/ac6b96/pdf

Bach, F. R., & Jordan, M. I. (2005). A probabilistic interpretation of canonical correlation analysis. University of California, Berkeley.

Ghahramani, Z., & Hinton, G. E. (1996). The EM algorithm for mixtures of factor analyzers. Technical Report CRG-TR-96-1, University of Toronto.

Gundersen, G. (n.d.). Probabilistic Canonical Correlation Analysis in Detail. Probabilistic Canonical Correlation Analysis in detail. https://gregorygundersen.com/blog/2018/09/10/pcca/ 

Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. Proceedings of the International Conference on Machine Learning.

Rubin, D. B., & Thayer, D. T. (1982). EM algorithms for ML factor analysis. Psychometrika, 47(1), 69-76. https://doi.org/10.1007/BF02293894

Williams, C. K. I., & Rasmussen, C. E. (2006). Gaussian processes for machine learning. MIT Press.

Zhao, Yuan, & Park, Il Memming. (2016). Variational Latent Gaussian Process for Recovering Single-Trial Dynamics from Population Spike Trains. Technical Report, Department of Neurobiology and Behavior, Department of Applied Mathematics and Statistics, Institute for Advanced Computational Sciences, Stony Brook, NY

https://github.com/catniplab/vlgp/tree/master

https://github.com/gwgundersen/ml/blob/master/probabilistic_canonical_correlation_analysis.py
