const express = require('express')
const mongoose  = require('mongoose')
const major_model = require('../models/major')
const faculty_model = require('../models/faculty')

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

module.exports.getMajorsByFaculties = function(req, res) {

}
