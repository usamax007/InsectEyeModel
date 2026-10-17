# The Insect Eye Model [1]
This repository contains code for an insect eye model that projects a **visual scene** onto the eye using gnomonic **projection** and subsequently detects motion within the projected scene using an event-based **motion detection** algorithm.

## Visual Scene Projection
The model computes a **projection of the visual scene** onto an array of photoreceptors in the hawkmoth *M. sexta*'s compound eye model. The compound eye is modeled as a simple hemispherical eye with a $180^\circ$ field of view along both azimuthal and elevation angles. Each facet connects to one photoreceptor projected by an interommatidial angle of $0.96^\circ$ [2,3]. The visual scene on a plane outside the eye is then projected onto the hemispherical array of photoreceptors using Gnomonic projection defined by the equations below [4].

$$\begin{align}
    x &= x_0 - \frac{mr_e}{\cos\theta}~{(\cos\phi_x\sin\phi_{x_0} - \sin\phi_x\cos\phi_{x_0})\cos(\phi_{y_0}- \phi_y)},\notag\\
    y &= y_0 + \frac{mr_e}{\cos\theta}~{\cos\phi_{x_0}\sin(\phi_{y_0}-\phi_y)},
\end{align}$$

where $x$ and $y$ are the coordinates of points on the image plane being projected from outside the eye, $x_0$ and $y_0$ are the coordinates of the center of focus of the hawkmoth's eye on the image plane (assumed (0,0)),

$$\begin{align}\theta = |\cos^{-1}(\sin\phi_x\sin\phi_{x_0} + \cos\phi_x\cos\phi_{x_0}\cos(\phi_{y_0} - \phi_y))|,\end{align}$$ 

$\phi_x$ and $\phi_y$ are the angular locations of a photoreceptor along the horizontal and vertical directions across the hemispherical photoreceptor array, $\phi_{x_0}$ and $\phi_{y_0}$ are values at the center of the array (assumed (0,0)), $r_e = 0.00205$ m is the radius of hawkmoth's eye [2], and $m = d_\text{mf}/r_e + 1$ is the magnification ratio where $d_\text{mf}$ is the distance between the hawkmoth and the center of focus on the plane.

Because the only moving object in the sample data is the robotic flower, the hawkmoth's field of view is restricted to $-80^\circ < \phi_x < 80^\circ$ and $-35^\circ < \phi_y < 35^\circ$ in order to discard the pixels with no information. The resulting grid is 36 pixels $\times$ 84 pixels, which makes a total of 3024 photoreceptors -- a small subset of $\sim25000$ facets on *M. sexta*'s compound eye [3]. The frames are sampled at 125 FPS, which is approximately the flicker fusion frequency of hawkmoths [5]. The luminosity level $L(p_x,p_y,n)$ of each pixel from the input frames is converted to the grayscale range 0-255, where $0 \leq p_x \leq 83$ and $0 \leq p_y \leq 35$ are the $x$ and $y$ pixel indices, respectively, and $n$ is the frame number.

## Event-based Motion Detection
To **detect flower motion**, we implement an **event-based motion detection** algorithm [6]. This simplified neuromorphic sensing technique is inspired by elementary motion detection in insect optic lobes [7]. At each pixel, it evaluates the luminosity difference between consecutive frames. If the absolute difference at a pixel $(p_x,p_y)$ between the $n$-th and $(n-1)$-th frame is greater than a certain threshold, an event $E(p_x,p_y,n)$ is said to be detected. A polarity of either $+1$ or $-1$ is assigned to that pixel depending on whether the luminosity difference was positive or negative, respectively. Otherwise, a value of $0$ is assigned. 

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

# Repository Code Description

The model processes an input video to generate two outputs:

1. **Visual Field (`output_vis.mp4`)** — Simulates the visual scene perceived through the compound eye by spatially integrating image intensities across individual ommatidia.
2. **Event Polarity (`output_epm.mp4`)** — Represents changes in logarithmic visual intensity between consecutive temporally sampled frames, producing positive and negative events based on a specified threshold.

The model currently supports anatomical and visual parameters of *Manduca sexta* (hawkmoth), including configurable angular field of view, interommatidial angles, gnomonic projection centers, and temporal sampling frequency.

## Repository Structure

```text
compound-eye-model/
│
├── src/
│   ├── main.py
│   └── compoundEyeModel.py
│
├── input/
│   └── flower_scn.mp4
│
├── output/
│   ├── output_vis.mp4
│   └── output_epm.mp4
│
└── README.md
```

- **`src/`** — Contains the Python source code for the compound-eye model and video processing.
- **`input/`** — Contains the input video files to be processed.
- **`output/`** — Stores the generated visual-field and event-polarity videos.
- **`README.md`** — Provides documentation and instructions for using the model.

The `output/` directory is automatically created if it does not already exist.

## Source Code Description

### 1. `main.py` — Video Processing Pipeline

The `main.py` script serves as the primary entry point for running the simulation. It manages video input/output, initializes the compound-eye model, and processes the input video to produce visual-field and event-polarity representations.

Its primary functions include:

- **Video input and output:** Reads the input video using OpenCV, extracts its resolution and frame rate, and initializes video writers for the two output representations.
- **Model configuration:** Specifies the insect species, angular field of view, gnomonic projection centers, and physical pixel scale.
- **Spatial processing:** Converts each input frame to grayscale and passes it to the `CompoundEyeModel` class to compute ommatidial visual intensities.
- **Temporal resampling:** Allows output videos to be generated at either the original input video FPS or the specified flicker-fusion frequency. Linear temporal interpolation is used when sampling times fall between input frames.
- **Event generation:** Computes the polarity of logarithmic intensity changes between successive output visual-field samples.
- **Performance monitoring:** Reports processing progress, execution time, and effective processing speed.

The main configuration parameters can be modified in the user-configuration section near the beginning of the script.

### 2. `compoundEyeModel.py` — Compound Eye Model

The `compoundEyeModel.py` file defines the `CompoundEyeModel` class, which implements the geometric and temporal components of the insect visual system.

**Spatial geometry and gnomonic projection**

The compound eye is represented as an angular grid of ommatidia distributed according to the specified interommatidial angles and total field of view.

The class implements the full gnomonic projection equations to map ommatidial angular coordinates onto a planar visual scene. Both projection-center angles are user-configurable.

Each projected ommatidial region is represented by a quadrilateral. Image pixels falling within each region are identified using a point-in-polygon algorithm, and their intensities are averaged to determine the corresponding ommatidial response.

To improve computational efficiency, the pixel-to-ommatidium mapping is calculated once and cached, provided the projection geometry, image dimensions, and spatial calibration remain unchanged.

**Temporal sampling and event polarity**

The model supports temporal sampling at either the input video's frame rate or a specified flicker-fusion frequency.

Visual-field samples at intermediate timestamps are generated using temporal interpolation. Event polarity is computed from the difference in logarithmic visual intensity between consecutive output samples.

The event-generation model supports:

- **Bipolar mode:** Positive events (255), negative events (0), and neutral responses (128).
- **Unipolar mode:** Events (255) and non-events (0).

An intensity-change threshold determines whether an event is generated.

## Installation and Requirements

The simulation requires Python 3 and the following libraries:

- NumPy — Numerical calculations and image intensity aggregation.
- OpenCV (`opencv-python`) — Video decoding, frame processing, and output encoding.

Install the dependencies using:

```bash
pip install numpy opencv-python
```

## Running the Simulation

**Step 1: Prepare the input video**

Place the video file inside the `input/` directory. By default, the script expects:

```text
input/flower_scn.mp4
```

To use a different video, update the corresponding input filename in `main.py`.

**Step 2: Configure the model**

Open `src/main.py` and modify the user-configuration parameters as needed.

For example, the angular field of view and projection centers can be configured as:

```python
SPECIES = "Ms"
ANGLE_X = 180.0
ANGLE_Y = 180.0

PHI_X_0 = 0.0
PHI_Y_0 = 0.0
```

The output temporal sampling rate can be selected using:

```python
FPS_MODE = "input"
```

or:

```python
FPS_MODE = "flicker_fusion"
FLICKER_FUSION_FPS = 125.0
```

In `"input"` mode, the output is sampled at the original video's frame rate. In `"flicker_fusion"` mode, the output is resampled at the specified frequency while approximately preserving the original video duration.

Note that temporal interpolation does not recover visual information absent from the input video, and the flicker-fusion frequency is used here as a configurable temporal sampling rate rather than a detailed physiological model of photoreceptor dynamics.

**Step 3: Execute the script**

From the repository root directory, run:

```bash
python src/main.py
```

The script resolves the input and output directories relative to its own location, so it does not require the current working directory to be `src/`.

**Step 4: Access the outputs**

After processing, the generated videos are saved in:

```text
output/
├── output_vis.mp4
└── output_epm.mp4
```

Both output videos use the selected temporal sampling rate and contain synchronized frames. The initial event-polarity frame is neutral because no preceding visual-field sample exists.

## Notes

- The gnomonic projection is valid only within its visible projection hemisphere; ommatidial regions outside that domain are excluded.
- The spatial pixel-to-ommatidium mapping is cached to avoid repeated geometric calculations for every video frame.
- The spatial integration and temporal sampling stages are implemented separately to reduce redundant computation.
- The generated MP4 videos are intended primarily for visualization. For quantitative analysis of event polarity or ommatidial intensity, lossless numerical data should be retained rather than relying on compressed video values.
