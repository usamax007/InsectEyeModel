# The Insect Eye Model [1]
This repository contains code for an insect eye model that projects a **visual scene** into the eye using gnomonic projection and then detects motion in the scene using an event-based motion detection algorithm.

The model computes a **projection of the visual scene** onto an array of photoreceptors in the hawkmoth *M. sexta*'s compound eye model. The compound eye is modeled as a simple hemispherical eye with a $180^\circ$ field of view along both azimuthal and elevation angles. Each facet connects to one photoreceptor projected by an interommatidial angle of $0.96^\circ$ [2,3]. The visual scene on a plane outside the eye is then projected onto the hemispherical array of photoreceptors using Gnomonic projection defined by the equations below [4].

$$\begin{align}
    x &= x_0 - \frac{mr_e}{\cos\theta}~{(\cos\phi_x\sin\phi_{x_0} - \sin\phi_x\cos\phi_{x_0})\cos(\phi_{y_0}- \phi_y)},\notag\\
    y &= y_0 + \frac{mr_e}{\cos\theta}~{\cos\phi_{x_0}\sin(\phi_{y_0}-\phi_y)},
\end{align}$$

where $x$ and $y$ are the coordinates of points on the image plane being projected from outside the eye, $x_0$ and $y_0$ are the coordinates of the center of focus of the hawkmoth's eye on the image plane (assumed (0,0)),

$$\begin{align}\theta = |\cos^{-1}(\sin\phi_x\sin\phi_{x_0} + \cos\phi_x\cos\phi_{x_0}\cos(\phi_{y_0} - \phi_y))|,\end{align}$$ 

$\phi_x$ and $\phi_y$ are the angular locations of a photoreceptor along the horizontal and vertical directions across the hemispherical photoreceptor array, $\phi_{x_0}$ and $\phi_{y_0}$ are values at the center of the array (assumed (0,0)), $r_e = 0.00205$ m is the radius of hawkmoth's eye [2], and $m = d_\text{mf}/r_e + 1$ is the magnification ratio where $d_\text{mf}$ is the distance between the hawkmoth and the center of focus on the plane.

Because the only moving object in the sample data is the robotic flower, the hawkmoth's field of view is restricted to $-80^\circ < \phi_x < 80^\circ$ and $-35^\circ < \phi_y < 35^\circ$ in order to discard the pixels with no information. The resulting grid is 36 pixels $\times$ 84 pixels, which makes a total of 3024 photoreceptors -- a small subset of $\sim25000$ facets on *M. sexta*'s compound eye [3]. The frames are sampled at 125 FPS, which is approximately the flicker fusion frequency of hawkmoths [5]. The luminosity level $L(p_x,p_y,n)$ of each pixel from the input frames is converted to the grayscale range 0-255, where $0 \leq p_x \leq 83$ and $0 \leq p_y \leq 35$ are the $x$ and $y$ pixel indices, respectively, and $n$ is the frame number.

To **detect flower motion**, we implement an **event-based motion detection** algorithm [6]. This simplified neuromorphic sensing technique is inspired by elementary motion detection in insect optic lobes [7]. At each pixel, it evaluates the luminosity difference between consecutive frames. If the absolute difference at a pixel $(p_x,p_y)$ between the $n$th and $(n-1)$th frame is greater than a certain threshold, an event $E(p_x,p_y,n)$ is said to be detected. A polarity of either $+1$ or $-1$ is assigned to that pixel depending on whether the luminosity difference was positive or negative, respectively. Otherwise, a value of $0$ is assigned. 

The following animation illustrates the model projecting a basic visual scene onto the eye and then detecting motion. 
<p align="center">
<img width="800" height="320" alt="gif_slides" src="https://github.com/user-attachments/assets/e1a2674e-446e-4e97-880d-bb0f8fd9475f" />
</p>

# References
1. U. B. Sikandar, H. Choi, J. Putney, H. Yang, S. Ferrari, S. Sponberg. "Predicting visually-modulated precisely-timed spikes across a coordinated and comprehensive motor program", Intl. Joint Conf. Neural Networks (IJCNN), 2023.
2. J. C. Theobald, E. J. Warrant, and D. C. O'Carroll, "Wide-field motion tuning in nocturnal hawkmoths," Proc. R. Soc. B Biol. Sci., vol. 277, no. 1683, pp. 853–860, 2009.
3. A. L. St¨ockl, W. A. Ribi, and E. J. Warrant, "Adaptations for nocturnal and diurnal vision in the hawkmoth lamina," J. Comp. Neurol., vol. 524, no. 1, pp. 160–175, 2016.
4. E. W. Weisstein, "Gnomonic projection." https://mathworld.wolfram.com/GnomonicProjection.html. Accessed: Jan 2023.
5. P. Chatterjee, U. Mohan, A. Krishnan, and S. P. Sane, "Evolutionary constraints on flicker fusion frequency in Lepidoptera," J. Comp. Physiol. A Neuroethol. Sensory, Neural, Behav. Physiol., vol. 206, no. 5, pp. 671–681, 2020.
6. J. Gemerek, "Active vision and perception," Ph.D. Thesis, 2020.
7. A. Bouzerdoum, "The elementary movement detection mechanism in insect vision," Philos. Trans. R. Soc. London. Ser. B Biol. Sci., vol. 339, no. 1290, pp. 375–384, 1993.
