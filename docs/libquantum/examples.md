# Examples using libquantum
This section showcases [] examples using various libquantum modules.

## Table of Contents

<!-- toc -->

- [Blast amplitude check](#blast-amplitude-check)
- [Blast CWT inverse](#blast-cwt-inverse)
- [Tone amplitude check](#tone-amplitude-check)
- [Sweep linear](#sweep-linear)
- [Sweep chirp](#sweep-chirp)
- [libquantum scales and export modules](#libquantum-scales-and-export-modules)


<!-- tocstop -->


### Blast amplitude check



In this example,...


![](img/example0.png)

**To run blast amplitude check example:**

In libquantum, inside examples folder: ```00_blast_amplitude_check.py```.
A copy of the code can be found [here]().

**libquantum modules used**: atoms, entropy, scales, spectra, utils, blast_pulse (as kaboom), 
plot_templates.plot_time_frequency_reps (as pltq).

### Blast CWT inverse

In this example,...

![](img/example1_1.png)

![](img/example1_2.png)

**To run blast cwt inverse example:**

In libquantum, inside examples folder: ```01_blast_cwt_inverse.py```.
A copy of the code can be found [here]().

**libquantum modules used**: [].

### Tone amplitude check

In this example,...

![](img/example2.png)

**To run tone amplitude check example:**

In libquantum, inside examples folder: ```02_tone_amplitude_check.py```.
A copy of the code can be found [here]().

**libquantum modules used**: atoms, entropy, scales, spectra, utils, synthetics, 
plot_templates.plot_time_frequency_reps (as pltq).


### Sweep linear

In this example,...

![](img/example3.png)

**To run sweep linear example:**

In libquantum, inside examples folder: ```03_sweep_linear.py```.
A copy of the code can be found [here]().

**libquantum modules used**: atoms, entropy, scales, spectra, utils, synthetics,
plot_templates.plot_time_frequency_reps (as pltq).

### Sweep chirp


**To run sweep chirp example:**

In libquantum, inside examples folder: ```04_sweep_chirp.py```.
A copy of the code can be found [here]().

**libquantum modules used**: atoms, entropy, scales, spectra, utils, synthetics,
plot_templates.plot_time_frequency_reps (as pltq), plot_templates.plot_time_frequency_picks (as pltpk).

### libquantum scales and export modules

The scales module constructs standardized scales, such as frequency band scales, that libquantum relies on. 
The export module exports time scales and frequencies to screen. This example showcases 
  

**To run libquantum scales and export example:**

In libquantum, inside examples folder: ```05_using_libquantum_scales.py```.
A copy of the code can be found [here]().

**libquantum modules used**: export, scales.
