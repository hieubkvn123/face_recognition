import React, { Component } from 'react'
import logo from './logo.svg';
import './App.css';
import Home from './components/Home'
import  {
	BrowserRouter as Router,
	Switch,
	Route,
	Redirect
} from 'react-router-dom'

import { LastLocationProvider } from 'react-router-last-location'

class App extends Component {
	constructor(props) {
		super(props)
	}

	render() {
		return (
			<div className="App">
				<Router>
					<LastLocationProvider>
						<Switch>
							<Route exact path='/' render={() => {
								return ( <Redirect to='home'/>)
							}}>
							</Route>
							<Route path="/home">
								<Home/>
							</Route>
						</Switch>
					</LastLocationProvider>
				</Router>
			</div>
		);
	}
}

export default App;
