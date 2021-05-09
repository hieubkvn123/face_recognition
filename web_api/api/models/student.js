const mongoose = require('mongoose')

const StudentSchema = mongoose.Schema({
	studentID : Number,
	firstName : String,
	lastName  : String,
	facultyID : String,
	majorName : String,
	imgPath : String
})

const StudentModel = mongoose.model("Student", StudentSchema)

module.exports = StudentModel