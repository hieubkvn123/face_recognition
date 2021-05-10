const mongoose = require('mongoose')

const FacultySchema = mongoose.Schema({
	facultyID : String,
	facultyName : String,
	facultyHead : String
})

const FacultyModel = mongoose.model("Faculty", FacultySchema)

module.exports = FacultyModel