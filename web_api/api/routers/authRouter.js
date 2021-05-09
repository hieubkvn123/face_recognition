const express = require('express')
const authController = require('../controllers/authController')

router = express.Router()
router.get('/get_faculties', authController.getAllFaculties)

module.exports = router