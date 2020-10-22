The main external dependency is autograd. Depending on your Python installation, you may have to use pip to install a few other standard packages (like multiprocessing and dill).

Most of the synthetic experiments in the paper can be run using the scripts in the experiments/ directory. We list the figure number along with the relevant script below; we describe how to construct Figure 1 and the real data experiments afterwards.

Figure 3: MnBoundPredError.py
Figure 4: hessAppxFigure.py
Figure 5: MnBoundPredErrorBad.py

To produce Figure 1, we use a script to compute exact CV for each of the three cases plotted. This can be replicated by running the three commands below. If you want to parallelize computation across folds of exact CV, you can use a --nCores option that is greater than zero.

python cmdExperiments.py --dataset SyntheticPoissonDatasetGenerator --model PoissonRegressionModel --regularization L2 --lambdaScaling const --lambdaCoeff 10.0 --upTo 30 --minNtrain 400 --maxNtrain 5000 --numNtrains 8 --NtoD const --D 40 --Xrank 40 --nCores 0

python cmdExperiments.py --dataset SyntheticPoissonDatasetGenerator --model PoissonRegressionModel --regularization L2 --lambdaScaling const --lambdaCoeff 10.0 --upTo 30 --minNtrain 400 --maxNtrain 5000 --numNtrains 8 --NtoD scaling --Xrank 40 --nCores 0

python cmdExperiments.py --dataset SyntheticPoissonDatasetGenerator --model PoissonRegressionModel --regularization L2 --lambdaScaling const --lambdaCoeff 10.0 --upTo 30 --minNtrain 400 --maxNtrain 5000 --numNtrains 8 --NtoD scaling --D 40 --nCores 0

Finally, run experiments/lowRankScalingPoisson.py to produce Figure 1.

To replicate the real data experiments, first download each dataset using the link in each class description in code/datasets.py; the class corresponding to each dataset is "BlogFeedbackDatasetGenerator" (Blog Feedback dataset) "RCV1DatasetGenerator" (RCV1 dataset), and "P53DatasetGenerator" (P53 dataset). Each downloaded dataset can be processed into text files using the scripts in the relevant data/ folders. Then the following commands can be run (again, to parallelize across folds of CV, specify an integer greater than zero for --nCores):

python cmdExperiments.py --dataset RCV1DatasetGenerator --modelName LogisticRegressionModel --regularization L2 --lambdaCoeff 50000 --lambdaScaling const --B 20 --nCores 0

python cmdExperiments.py --dataset P53DatasetGenerator --modelName LogisticRegressionModel --regularization L2 --lambdaCoeff 20000 --lambdaScaling const --B 20 --nCores 0

python cmdExperiments.py --dataset BlogFeedbackDatasetGenerator --modelName PoissonRegressionModel --regularization L2 --lambdaCoeff 50000 --lambdaScaling const --B 20 --nCores 0

The script experiments/realDataExperiments.py then processes the exact CV runs and produces the data needed to produce Figure 2; the file will be saved as experiments/realDataResults.pkl (note if you changed any options in the above commands, you may have to change the filename of the outputs in realDataExperiments.py). As producing the exact CV runs and running realDataExperiments.py is very time consuming, we have included the version of this file used to make Figure 2. To produce Figure 2, run experiments/realDataPlots.py.






