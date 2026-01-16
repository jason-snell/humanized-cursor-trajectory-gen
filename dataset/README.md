This dataset contains ~8000 cursor movement trajectories collected by manually clicking a button in a web page.

## Collection Method

A button was randomly placed on a webpage. When clicked, it moved to a new random location. Cursor movements between clicks were logged, creating trajectories from the previous button position to the next.

Video demonstrating the data collection interface, with the trained model automating the clicks
![Demo](demo.mp4)

## Data Format

### Fields

- **id**: Unique identifier for each movement
- **type**: Input device (0 = mouse, 1 = trackpad)
- **screen**: Display dimensions (`w` = width, `h` = height in pixels)
- **start**: Starting coordinates [x, y]
- **end**: Ending coordinates [x, y]
- **trajectory**: Array of [x, y] coordinates representing the cursor path
- **time**: Timestamp of the movement

### Available Formats

- **JSON**: Full structured format with all fields
- **CSV**: Compact format with semicolon-separated trajectory points
- **MySQL**: Original database format with pipe-separated trajectory points

### Example

```json
{
  "id": 15,
  "type": 0,
  "screen": {"w": 1920, "h": 941},
  "start": [1007, 629],
  "end": [816, 470],
  "trajectory": [[1007,629], [1009,623], ..., [816,470]],
  "time": "2023-10-09 03:02:05"
}
```
