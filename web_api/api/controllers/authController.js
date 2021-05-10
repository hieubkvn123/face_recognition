const express = require('express')
const mongoose  = require('mongoose')
const major_model = require('../models/major')
const faculty_model = require('../models/faculty')
const student_model = require('../models/student')
const user_model = require('../models/user')

/* Connect to MongoDB */
mongoose.connect("mongodb://localhost:27017/face_recog", {
	useNewUrlParser : true,
	useUnifiedTopology : true
})

/* Controllers */
module.exports.getAllFaculties = function(req, res) {
	faculty_model.find(function(err, docs) {	
		if(err) {
			throw err
		}

		res.send(docs)
	})
}

module.exports.getMajorsByFaculty = function(req, res) {
	var facultyID = req.body['facultyID']

	major_model.find({facultyID : facultyID}, function(err, docs) {
		if(err) {
			throw err
		}

		res.send(docs)
	})
}

module.exports.register = async function(req, res) {
	var formData = req.body 

	var student_data = {
		studentID : formData['studentID'],
		firstName : formData['firstName'],
		lastName  : formData['lastName'],
		facultyID : formData['facultyID'],
		majorName : formData['majorName'],
		imgPath : null
	}

	var user_data = {
		studentID : formData['studentID'],
		userName : formData['userName'],
		password : formData['password']
	}

	// Need to check if data already exist first

	console.log('[INFO] Inserting into student collection...')
	await student_model.collection.insertOne(student_data, function(err, docs) {
		if(err) {
			throw err
		}
	})

	console.log('[INFO] Inserting into user collection...')
	await user_model.collection.insertOne(user_data, function(err, docs) {
		if(err) {
			throw err
		}

		res.send('success')
	})
}

