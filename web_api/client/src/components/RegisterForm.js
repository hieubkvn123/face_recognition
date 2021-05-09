import React, { Component } from "react";
import axios from 'axios'

import config from '../assets/config.json'
import "../styles/register_form.css";
import 'bootstrap/dist/css/bootstrap.min.css'

class RegisterForm extends Component {
  constructor(props) {
    super(props)
    this.state = {
      studentID : '',
      username : '',
      firstName : '',
      lastName : '',
      facultyID : '',
      majorName : '',
      password : '',
      all_faculties : [],
      majors_by_faculty : []
    }

    /* Bind the handlers */
    this.render = this.render.bind(this)
    this.componentDidMount = this.componentDidMount.bind(this)
    this.onFieldChange = this.onFieldChange.bind(this)
    this.validateForm = this.validateForm.bind(this)
    this.handleSubmit = this.handleSubmit.bind(this)
  }

  componentDidMount = async function() {
    await axios({
      'method' : 'post',
      'url' : `http://${config['api_host']}:${config['api_port']}/auth/get_faculties`,
      'headers' : {
        'Content-Type' : 'multipart/form-data'
      }
    }).then(response => this.setState({all_faculties : response.data}))
    .catch(err => console.log(err))


    // Get all majors
    await axios({
      'method' : 'post',
      'data' : {'facultyID' : this.state.all_faculties[0]['facultyID']},
      'url' : `http://${config['api_host']}:${config['api_port']}/auth/get_majors_by_faculty`,
      'headers' : {
        'Content-Type' : 'application/json'
      }
    }).then(response => this.setState({majors_by_faculty : response.data}))
    .catch(err => console.log(err))

    console.log(this.state.all_faculties)
  }

  validateForm() {
    // Student ID must be all number
    // Username, First Name, Last Name and Password must not be empty
    return this.state.username.length > 0 && this.state.password.length > 0;
  }

  handleSubmit(event) {
    event.preventDefault();

    var data = {
      username : this.state.username,
      firstName : this.state.firstName,
      lastName : this.state.lastName,
      facultyID : this.state.facultyID,
      majorName : this.state.majorName,
      password : this.state.password,
    }

    if(this.validateForm()){
      axios({
        'method' : 'post',
        'data' : data,
        'url' : `http://${config['api_host']}:${config['api_port']}/auth/register`,
        'headers' : {
          'Content-Type' : 'application/json'
        }
      }).then(response => {
        if(response.data == 'success') 
          alert('Registration success')
      })
      .catch(err => console.log(err))
    }else{
      alert("Please fill out the form")
    }
  }

  onFieldChange(event) {
    this.setState({
      [event.target.name] : event.target.value
    })

    console.log(event.target.name)

    if(event.target.name == 'facultyID') {
      axios({
        'method' : 'post',
        'data' : {'facultyID' : event.target.value},
        'url' : `http://${config['api_host']}:${config['api_port']}/auth/get_majors_by_faculty`,
        'headers' : {
          'Content-Type' : 'application/json'
        }
      }).then(response => this.setState({majors_by_faculty : response.data}))
      .catch(err => console.log(err))
    }
  }

  render(){
    return (
      <div className="Register">
        <form className='form-container'>
          <h1>Register</h1>
        {/* StudentID */}
          <div className="form-group">
            <label className='register-label' htmlFor='register-studentid'>StudentID</label>
            <input name='studentID' id='register-studentid' type="text" className="form-control" placeholder="Enter student ID" onChange={this.onFieldChange}/>
          </div>

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
            <label className='register-label' htmlFor='register-faculty'>Faculty</label>
            <select name='facultyID' id='register-faculty' className="form-control" onChange={this.onFieldChange}>
              {
                this.state.all_faculties.map((value, index) => {
                  if(index == 0)
                    return (<option default selected value={value.facultyID}>{value.facultyID}</option>)
                  else
                    return (<option value={value.facultyID}>{value.facultyID}</option>)
                })
              }
            </select>
          </div>

          {/* Major Name */}
          <div className="form-group">
            <label className='register-label' htmlFor='register-major'>Major</label>
            <select name='majorName' id='register-major' className="form-control" onChange={this.onFieldChange}>
              {
                this.state.majors_by_faculty.map((value, index) => {
                  if(index == 0)
                    return (<option default selected value={value.majorName}>{value.majorName}</option>)
                  else
                    return (<option value={value.majorName}>{value.majorName}</option>)
                })
              }
            </select>
          </div>

          {/* Password */}
          <div className="form-group">
            <label className='label' htmlFor='register-password'>Password</label>
            <input name='password' id='register-password' type="password" className="form-control" placeholder="Enter password" onChange={this.onFieldChange}/>
          </div>

          <button type="submit" className="btn btn-dark btn-lg btn-block" onClick={this.handleSubmit}>Sign up</button>
        </form>
      </div>
    );
  }
}

export default RegisterForm