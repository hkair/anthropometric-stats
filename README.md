## Anthropometric Stats
A plotly dash application for exploring the ANSUR II dataset on human anthropometric measurements. The measurements or variables are all shown in http://tools.openlab.psu.edu/publicData/ANSURII-TR15-007.pdf.
It's main functionality include the display of the distribution of measurements based on its subset sub-population (gender, race, height). Summary Statistics
such as the mean, standard error, standard deviation and more are shown.

## Motivation
The problem I wanted to figure out arose from trying to improve my squat and deadlift form. More specifically, how can I increase my depth during the squat? The depth of a squat refers to how low one's hip gets during the squat.
![Depth](https://i.ytimg.com/vi/7cWgc4q7pxg/maxresdefault.jpg)

I came across multiple viewpoints on why some people are not able to reach a certain depth no matter how hard they try. It is clear that body proportion is very important in the sport of powerlifting. For example people with longer femur/tibia (Thigh/Calf) ratios have a more difficult time reaching depth because of the longer moment arm between the hip and the knee as the pivot. The same logic applies to the torso, where people with larger torso has a larger moment arm at the hip!

So, my next question was to find out the average body measurements of the torso, femur and tibia in the human population. If we were to find these measurements within
each sub-population, we are able to figure out how each of us place within the population. If you are a male, 176cm tall and a certain race, how well to compare to your "sub-population" in terms of body proportions? 

After browsing the internet, I came across the ANSUR II dataset which contains innumerable number of measurements of the human body. All the way from Abdominal Extension Sitting to the bitragion sub-mandibular arc measurement (length afrom eat to eat around the jaw). The measurements were provided by the US Army in 2012, so it comes to no surprise of the standardization of each measurements. There were in total 6068 participants, with 4082 males and 1986 females.

## Screenshots

### Front Page
![Page](https://i.imgur.com/8fEorTV.png)

### Distribution Plot of Abdominal Extension Depth, Sitting
![distribution plot](https://i.imgur.com/tqsoNQj.png)

### Body Skeleton from Proportion Constants
![body-skeleton plot](https://i.imgur.com/sICmsV7.png)

### Measurement Image 
![measurement-img](https://i.imgur.com/dPvbeYE.png)

### Lebron James and Pose Estimation
![pose-estimation and body ratio](https://i.imgur.com/oLg6vE0.png)

## Tech/framework used

<b>Built with</b>
- [Plotly]([https://plotly.com/dash/](https://plotly.com/dash/))
- [Opencv]([https://opencv.org/](https://opencv.org/))
- [mediapipe]([https://mediapipe.dev/](https://mediapipe.dev/))

## Features
Alongside the various plotly plots, the dash application allows you to input a single image of a human standing relatively vertical across the floor and calculates the body
proportion ratio from the pose estimation. For this project, the mediapipe pose-estimation model was used. The body proportion calculations from the poses are contained
in the poseModule.py. 

## Installation

1. Clone the project
``` 
git clone https://github.com/hkair/anthropometric-stats.git
```

2. Activate the Environment
```
source env/bin/activate
```

3. Install Dependencies from requirements.txt
```
pip install requirements.txt 
```

4. Run ansur_dash.py
``` 
python ansur_dash.py
```

## How to use?
Go to your local host or visit https://anthropometric-stats-dash.herokuapp.com/! And play around! See how you place amongst your sub-population.
Note that some races may not be represented accurately becuase of the low number of participants, for example there are only 188 asians and 49 Native Americans
in the ANSUR II Dataset, so interpret the data however you will. 

## Credits
Many thanks to the US Army for creating this wonderful dataset and the team at penn state for providing this dataset https://www.openlab.psu.edu/ansur2/.

## Licence
Licensed under the [MIT LICENSE](LICENSE).
