import React, { Component } from "react";
import LoginForm from "./LoginForm";
import RegisterForm from './RegisterForm'
import { 
	BrowserRouter as Router, 
	Switch, 
	Route, 
	Redirect,
	useRouteMatch
} from "react-router-dom";
import {LastLocationProvider} from 'react-router-last-location'

import Navbar from 'react-bootstrap/Navbar'
import {
	Nav, Button
} from 'react-bootstrap'

/* Any CSS goes here */
import '../styles/home.css'

function Home() {
	let { path, url } = useRouteMatch()

	return (
		<div className="home-container" style={{'backgroundImage':`url('${process.env.PUBLIC_URL}/home_background.jpg')`}}>
			{/* The Navigation Bar */}
			<Navbar bg='dark' variant='dark' id='home-nav-bar'>
				<Navbar.Brand  href={`${path}/login`}>
					<img style={{'height':'50px', 'position':'relative'}} src={`${process.env.PUBLIC_URL}/uow_logo.png`}/>
				</Navbar.Brand>
				<Nav>
					<Nav.Link href={`${path}/login`}>Login</Nav.Link>
					<Nav.Link href={`${path}/register`}>Register</Nav.Link>
				</Nav>
			</Navbar>

			{/* The Router */}
			<Router>
				<LastLocationProvider>
					<Switch>
						<Route path={`${path}/login`}>
							<LoginForm/>
						</Route>
						<Route exact path={`${path}/register`}>
							<RegisterForm/>
						</Route>
						<Route exact path={`${path}/`} render={()=>{
							return (<Redirect to={`${path}/login`}/>) 
						}}/>
					</Switch>
				</LastLocationProvider>
			</Router>
		</div>
	);
}

export default Home;

