# FTIR_analyzer
___________________________________________________________________________________
__author__ = Dr. *Giada Innocenti* \
__credits__ = Dr. *Juliana Silva Alves Carneiro*, *Bryan J. Hare* for reporting some bugs.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
*Flaherty's Laboratory* for coding the OG PSD part of this script.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[*NIRPY Research*](https://nirpyresearch.com) for the MSC code.\
__email__ = giadainnocenti.gi@gmail.com \
__Python__ = 3.8.5
__________________________________________________________________________________

This script is able to plot, and analyze Fourier Transform Infrared Spectroscopy (FTIR) data. It relies on [scipy](https://www.scipy.org/scipylib/index.html), [lmfit](https://lmfit.github.io/lmfit-py/intro.html), [pymcr](https://pages.nist.gov/pyMCR/). 

### What can FTIR analyzer do?
1. Read and subtract the CSV catalyst spectrum from the catalyst + molecule and plot the results.
2. Work with data collected with respect to time on stream (days, h, min, s) or with respect to temperature (℃ or K).
3. Remove outliers due to instrumental noise (the spectrum increase over a certain threshold)
4. Remove all the wavenumber that do not have any information, usually over 4000 and lower than 1000 cm<sup>-1</sup>.
5. Remove the baseline by using the Asymmetric Least Square Smoothing with the most commonly used default settings.
6. Find the peaks on your spectrum and plot their wavenumber on the figure
7. Subtract a specific temperature or time spectrum from other spectra of your choice and save the results in CSV files.
8. Calculate the first and second derivative of your spectra.
9. Deconvolute your peaks of interest by using many different distribution shapes or combination of them and it will plot the obtained trends with respect to the time or temperature. CSV file will be generated containing both the integrations data and the IR data to reproduce the plot using another software (Origin, excel ...).
10. Analyze excitation emission spectroscopy IR experiments that use a synusoidal perturbation.

A detailed Explanation of the script and of all the keys is reported into the jupyter notebook.


Future implementations (that may or may not be  added by the author):

1. it would be nice being able to implement the atmospheric suppression into the code. It should be failrly easy to code. It would be necessary to get the H<sub>2</sub>O vapor and the CO<sub>2</sub> vapor spectra from the NIST. Make sure that the number of data points for those spectra is the same than the experimental spectra. Convolute the CO<sub>2</sub> and H<sub>2</sub>O spectra by using a linear combination. minimize such convolution and then removed the minimized H<sub>2</sub>O and CO<sub>2</sub> spectrum. I think this is also what omnic does when it remove H<sub>2</sub>O and CO<sub>2</sub> with atmospheric suppression.
2. Adding the possibility of analyzing MES-PSD data that use a non synusoidal perturbation.
3. Converting the Jupyter Notebook into a web-app working on a local machine (not online for data safety purposes).
