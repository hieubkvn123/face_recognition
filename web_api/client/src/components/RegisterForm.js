import React, { Component } from "react";
import "../styles/register_form.css";
import 'bootstrap/dist/css/bootstrap.min.css'

class RegisterForm extends Component {
  constructor(props) {
    super(props)
    this.state = {
      username : '',
      firstName : '',
      lastName : '',
      facultyID : '',
      majorName : '',
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
      <div className="Register">
        <form className='form-container'>
          <h1>Register</h1>

          {/* Username */}
          <div className="form-group">
            <label className='register-label' htmlFor='register-username'>Username</label>
            <input name='username' id='register-username' type="text" className="form-control" placeholder="Enter username" onChange={this.onFieldChange}/>
          </div>

          {/* First Name */}
          <div className="form-group">
            <label className='register-label' htmlFor='register-fname'>First Name</label>
            <input name='firstName' id='register-fname' type="text" className="form-control" placeholder="Enter First Name" onChange={this.onFieldChange}/>
          </div>

          {/* LastName */}
          <div className="form-group">
            <label className='register-label' htmlFor='register-lname'>Last Name</label>
            <input name='lastName' id='register-lname' type="text" className="form-control" placeholder="Enter LastName" onChange={this.onFieldChange}/>
          </div>

          {/* Faculty ID */}
          <div className="form-group">
            <label className='register-label' htmlFor='register-username'>Faculty</label>
            <input name='username' id='register-username' type="text" className="form-control" placeholder="Enter username" onChange={this.onFieldChange}/>
          </div>

          {/* Major Name */}
          <div className="form-group">
            <label className='register-label' htmlFor='register-username'>Major</label>
            <input name='username' id='register-username' type="text" className="form-control" placeholder="Enter username" onChange={this.onFieldChange}/>
          </div>

          {/* Password */}
          <div className="form-group">
            <label className='label' htmlFor='register-password'>Password</label>
            <input name='password' id='register-password' type="password" className="form-control" placeholder="Enter password" onChange={this.onFieldChange}/>
          </div>

          <button type="submit" className="btn btn-dark btn-lg btn-block">Sign up</button>
        </form>
      </div>
    );
  }
}

export default RegisterForm