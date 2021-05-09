const mongoose = require('mongoose')
const faculty = require('../models/faculty')
const major = require('../models/major')

/* Connect to MongoDB */
mongoose.connect("mongodb://localhost:27017/face_recog", {
	useNewUrlParser : true,
	useUnifiedTopology : true
})

/* Remove old data first */
console.log('[INFO] Removing old data ... ');
var deleteAllRecords = (async function() {
	await faculty.collection.deleteMany({})
	await major.collection.deleteMany({})

	return 'success'
});

const faculties = [
	new faculty({facultyID : 'SoCIT', facultyName : 'School of Computing and Information Technology', facultyHead : 'WillySusilo'}),
	new faculty({facultyID : 'MAT01', facultyName : 'School of Applied Mathematics', facultyHead : 'SeanGoodfellow'}),
	new faculty({facultyID : 'MAT02', facultyName : 'School of Engineering Mathematics', facultyHead : 'IanNg'}),
	new faculty({facultyID : 'BIO01', facultyName : 'School of Biomedical Engineering', facultyHead : 'SuzieElise'}),
	new faculty({facultyID : 'CSE01', facultyName : 'School of Computer Engineering', facultyHead : 'MaiDacCuong'})	
];

const majors = [
	new major({facultyID : 'SoCIT', majorName : 'Computer Science (Big Data)'}),
	new major({facultyID : 'SoCIT', majorName : 'Computer Science (Mobile and Game Development)'}),
	new major({facultyID : 'SoCIT', majorName : 'Computer Science (Digital System Security)'}),
	new major({facultyID : 'SoCIT', majorName : 'Computer Science (Software Engineering)'}),
	new major({facultyID : 'SoCIT', majorName : 'Computer Science (Cyber Security)'}),

	new major({facultyID : 'MAT01', majorName : 'Data Science and Business Analytics'}),
	new major({facultyID : 'MAT01', majorName : 'Data Science and Machine Learning'}),
	new major({facultyID : 'MAT01', majorName : 'Finacial Mathematics and Business Analytics'}),
	new major({facultyID : 'MAT01', majorName : 'Mathematics and Economics'}),

	new major({facultyID : 'MAT02', majorName : 'Engineering Mathematics (Petrolium)'}),
	new major({facultyID : 'MAT02', majorName : 'Engineering Mathematics (Electrical Engineering)'}),
	new major({facultyID : 'MAT02', majorName : 'Engineering Mathematics (Mechanical Engineering)'}),

	new major({facultyID : 'BIO01', majorName : 'Biomedical Engineering (Biological system prosthetics)'}),
	new major({facultyID : 'BIO01', majorName : 'Biomedical Engineering (Bioinformatics)'}),
	new major({facultyID : 'BIO01', majorName : 'Biomedical Engineering (Computational biology robotics)'}),

	new major({facultyID : 'CSE01', majorName : 'Computer Engineering'}),
	new major({facultyID : 'BIO01', majorName : 'Computing and information system'}),
	new major({facultyID : 'BIO01', majorName : 'Computing and Electrical Engineering'})
];

var insertRecordsFaculty = (async function() {
	await faculty.collection.insertMany(faculties, function(err, docs) {
		if(err) { throw err }
		console.log(`[INFO] Inserted ${docs.length} documents into Faculty ... `)
	})

	return 'success'
})

var insertRecordsMajor = (async function() {	
	await major.collection.insertMany(majors, function(err, docs) {
		if(err) { throw err }
		console.log(`[INFO] Inserted ${docs.length} documents into Major ... `) 
	})

	return 'success'
});

var exit = (async function() {
	process.exit()
})

var ahihi = deleteAllRecords()
	.then(() => insertRecordsFaculty())
	.then(() => insertRecordsMajor())

setTimeout(() => (process.exit()), 3000)