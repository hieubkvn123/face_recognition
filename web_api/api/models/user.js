const mongoose = require('mongoose')

const UserSchema = mongoose.Schema({
	studentID : Number,
	userName : String,
	password : String
})

const UserModel = mongoose.model("User", UserSchema)

module.exports = UserModel