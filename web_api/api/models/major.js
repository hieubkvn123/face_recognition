const mongoose = require('mongoose')

const MajorSchema = mongoose.Schema({
	facultyID : String,
	majorName : String
})

const MajorModel = mongoose.model("Major", MajorSchema)

module.exports = MajorModel