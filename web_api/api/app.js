const express = require('express')
const cors = require('cors')
const config = require('./config/default')

/* Routers */
const autRouter = require('./routers/authRouter')

app = express()
app.use(cors())

/* Register routers */
app.use('/auth', autRouter)

app.get('/', (req, res) => {
	res.send("Hello")
})

app.listen(config['port'], () => {
	console.log(`[INFO] Server is listening on port ${config['port']}`)
})