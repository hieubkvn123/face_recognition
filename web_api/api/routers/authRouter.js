const express = require('express')
const bodyParser = require('body-parser')
const authController = require('../controllers/authController')

router = express.Router()

// parse application/x-www-form-urlencoded
router.use(bodyParser.urlencoded({ extended: false }))
// parse application/x-www-form-urlencoded
router.use(bodyParser.json())

/* Registering controllers */
router.post('/get_faculties', authController.getAllFaculties)
router.post('/get_majors_by_faculty', authController.getMajorsByFaculty)
router.post('/register', authController.register)

module.exports = router