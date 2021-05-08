import React, { Component } from "react";
import "../styles/login_form.css";
import 'bootstrap/dist/css/bootstrap.min.css'

class LoginForm extends Component {
  constructor(props) {
    super(props)
    this.state = {
      username : '',
      password : ''
    }

    /* Bind the handlers */
    this.onFieldChange = this.onFieldChange.bind(this)
    this.validateForm = this.validateForm.bind(this)
    this.handleSubmit = this.handleSubmit.bind(this)
  }

  validateForm() {
    return this.state.username.length > 0 && this.state.password.length > 0;
  }

  handleSubmit(event) {
    event.preventDefault();
  }

  onFieldChange(event) {
    this.setState({
      [event.target.name] : event.target.value
    })
  }

  render(){
    return (
      <div className="Login">
        <form className='form-container'>
          <h1>Log in</h1>

          <div className="form-group">
            <label className='login-label' htmlFor='login-username'>Username</label>
            <input name='username' id='login-username' type="text" className="form-control" placeholder="Enter username" onChange={this.onFieldChange}/>
          </div>

          <div className="form-group">
            <label className='label' htmlFor='login-password'>Password</label>
            <input name='password' id='login-password' type="password" className="form-control" placeholder="Enter password" onChange={this.onFieldChange}/>
          </div>

          <button type="submit" className="btn btn-dark btn-lg btn-block">Sign in</button>
        </form>
      </div>
    );
  }
}

export default LoginForm