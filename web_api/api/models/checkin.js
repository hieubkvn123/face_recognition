const mongoose = require('mongoose')

const CheckinSchema = mongoose.Schema({
	userName : String,
	checkinDateTime : Date
})

const CheckinModel = mongoose.model("Checkin", CheckinSchema)

module.exports = CheckinModel