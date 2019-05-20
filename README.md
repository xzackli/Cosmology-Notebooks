# Cosmology Notebooks

## CAMBvsCLASS
This is a quick and dirty Jupyter notebook that compares reference spectra output for CLASS and CAMB. At
the highest precision settings, they agree at a 0.1% level up to $\ell \sim 3500$. The linear matter
power spectra agree to ~0.01%, and HALOFIT nonlinear P(k) agree up to ~0.1%.

I'm particularly interested in the TT spectrum for $3000 < \ell < 5000$. For these purposes, the key parameter appears
to be `k_max_tau0_over_l_max`.

https://colab.research.google.com/drive/1zLWfk3qK6qRyPT2HZhSUQarCcQqGALCa

https://docs.google.com/presentation/d/1k1CAUu-_QdhZXCH6k9iAlionUD00OpiZgFRnOgO7nIk/edit?usp=sharing
