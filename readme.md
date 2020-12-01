# Codes for the talk ``Nonnegative and Low-Rank Approximations''

These codes can be used to reproduce the experiments on the talk ``Nonnegative and Low-Rank Approximations'', provided in this repo as well.

## Requirements:
- numpy
- scikit-learn
- nn_fac (version 0.1.2 and more)
- matplotlib
- scipy
Install by running pip install --user -r requirements.txt or any other similar command you like.

## Description of the files
- fluorescence.py: use Nonnegative Tensor Factorization to find hidden chemical spectra and relative concentrations. Data is a courtesy of Rasmus Bro, check [http://www.models.life.ku.dk/Amino_Acid_fluo](http://www.models.life.ku.dk/Amino_Acid_fluo) and credit ``Bro, R, PARAFAC: Tutorial and applications, Chemometrics and Intelligent Laboratory Systems, 1997, 38, 149-171'' for any external use.
- text_mining.py: have fun exploring the content of text files of your choice with Nonnegative Matrix Factorization. Put .txt files in the ./textes folders with various topics and watch how NMF recovers topics and text clusters!
- transcription.py: a toy example of automatic transcription from audio to a tentative collection of notes and activations, using NMF. It does not work tremendously well, but you can try with your own audio file and see how it goes!
- homer_2color.py: recolor a homer picture with 2 colors of your choice, using Nonnegative Least Squares.
- honer_ncolor.py: recolor a homer picture with more colors (here 5) again with nnls, and showing sparsity levels.
- homer_nmf: Computing a rank-2 version of a homer image with given colors, and then trying to find thoses colors from the image only using exact rank2 NMF.
