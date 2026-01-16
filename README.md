### **Humanized Trajectory Generator**

Human-like cursor movement generator using a self-trained ONNX model. Simulates real mouse trajectories for testing, UI automation, or research.

Generates intermediate points between a start X,Y and end X,Y. Useful against anti-bots for automation.

I trained this late 2023 on a dataset of my own cursor movement. If I come across the data, I will add here.

Video demonstrating the data collection interface, with the trained model automating the clicks (https://www.youtube.com/watch?v=eyEzAjEbgxI)


https://github.com/user-attachments/assets/84be3cda-3fd4-42cc-bd58-e943f99d0f66

There are two separate projects.

`TrajectoryGeneratorAPI` is an API implementation, which can be tested here: https://jsnell.dev/trajectory/

```
curl -s -L -X POST 'https://jsnell.dev/api/generate' ^
-H 'Content-Type: application/json' ^
-d '{
	"start": [
		903,
		629
	],
	"end": [
		1021,
		726
	],
	"points": 15,
	"randomness": 1
}'

// Response
{
	"points": [
		[
			903,
			629
		],
		[
			889,
			622
		],
    ...
		[
			1021,
			726
		]
	]
}
```

<img width="1862" height="890" alt="chrome_KIslIPYKBx" src="https://github.com/user-attachments/assets/84dd3adb-6867-4768-88b0-0ef72c8dc3f5" />


`HumanizedTrajectoryGen` is a very basic CLI example:

```
Enter nothing for randomized data:
Enter start point (x,y):
No input provided. Generating random coordinates: 162,584
Enter end point (x,y):
No input provided. Generating random coordinates: 1396,9
Enter randomness factor (optional, default: 1.5):  (Default: 1.5)
Invalid input. Please enter a number.
Enter density (optional, default: 5):  (Default: 5)
Value must be at least 1. Using default.
Trajectory:
[[162,584],[164,584],[167,584],[170,584],[173,583],..]
Image saved successfully to output\output-01-56-09.png
```

![output-01-56-09](https://github.com/user-attachments/assets/ed01fcef-4825-4460-bb8b-6eed7afadf8b)
