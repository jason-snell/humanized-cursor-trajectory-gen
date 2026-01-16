const fs = require('fs');

if (!fs.existsSync('dataset.sql')) {
	console.log('dataset.sql not found');
	return;
}

const lines = fs.readFileSync('dataset.sql', 'utf-8').split('\r\n');

const pattern = /^\((?<id>\d+),\s(?<type>\d+),\s'(?<w>\d+),(?<h>\d+)',\s'(?<points>[\d+,\|]+)',\s'(?<time>[\d+\-\s:]+)'\)[;,]?$/mg;

const csv = ['id,type,screen_w,screen_h,"trajectory","timestamp"'];
const objs = [];

for (let i = 0; i < lines.length; i++) {
	for (const match of lines[i].matchAll(pattern)) {
		const {id, type, w, h, points, time} = match.groups;

		const trajectory = points.split('|');

		csv.push(`${id},${type},${w},${h},"${trajectory.join(';')}","${time}"`);
		objs.push({
			id: +id,
			type: +type,
			screen: {
				w: +w,
				h: +h
			},
			start: `[${trajectory[0]}]`,
			end: `[${trajectory[trajectory.length - 1]}]`,
			trajectory: `[${trajectory.map(x => `[${x.split(',')}]`)}]`,
			time: time
		});
	}
}

fs.writeFileSync('trajectory.csv', csv.join('\n'));
console.log('wrote trajectory.csv');

const parsedJson = JSON.stringify(objs, null, 2).replace(/"\[/g, '[').replace(/\]"/g, ']');

fs.writeFileSync('trajectory.json', parsedJson);
console.log('wrote trajectory.json');