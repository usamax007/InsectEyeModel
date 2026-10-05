# InsectEyeModel
This repository contains code for a hawkmoth eye model that projects a visual scene into the eye using gnomonic projection and then detects motion in the scene using an event-based motion detection algorithm.

The model computes the projection of a visual scene onto an array of photoreceptors in the hawkmoth’s compound eye model. The compound eye is modeled as a simple hemispherical eye with a $180^\circ$ field of view along both azimuthal and elevation angles. Each facet connects to one photoreceptor projected by an interommatidial angle of $0.96^\circ$. The visual scene of the flower on a plane outside the eye is then projected onto the hemispherical array of photoreceptors using Gnomonic projection.

To detect flower motion, an event-based motion detection algorithm is implemented. This is a simplified neuromorphic sensing technique inspired by elementary motion detection in the optic lobes of insects. At each pixel, it evaluates the luminosity difference between consecutive frames. A polarity of either $+1$ or $−1$ is assigned to that pixel depending on whether the luminosity difference was positive or negative, respectively. Otherwise, a value of $0$ is assigned.

