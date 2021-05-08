const express = require('express')
const cors = require('cors')
const config = require('./config/default')

app = express()
app.use(cors())

app.get('/', (req, res) => {
	res.send("Hello")
})

app.listen(config['port'], () => {
	console.log(`[INFO] Server is listening on port ${config['port']}`)
})