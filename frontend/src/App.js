import "./styles.css";
import React, { useState } from "react";
import styled from "styled-components";
import axios from "axios";

const StyledInput = styled.input`
  display: block;
  margin: 20px 0px;
  font-size: 18px;
  padding: 8px;
  border: 1px solid lightblue;
`;

function useInput(defaultValue = "") {
  const [value, setValue] = useState(defaultValue);
  function onChange(e) {
    setValue(e.target.value);
  }
  return {
    value,
    onChange,
  };
}

export default function App() {
  // Using a fancy shared React hook to control input state
  const inputProps = useInput();
  const [userList, setUserList] = useState([]);
  const [loading, setLoading] = useState(false);
  // Using native DOM form submission to store state
  function submitForm(event) {
    // Prevent the default event because it causes a page refresh
    event.preventDefault();
    const data = new FormData(event.target);
    alert(data.get("myInput"));
  }
  function useInput(defaultValue = "") {
    const [value, setValue] = useState(defaultValue);
    function onChange(e) {
      setValue(e.target.value);
    }
    return {
      value,
      onChange,
    };
  }
  return (
    <div>
      <h1> React Input Examples </h1>
      <h2> Uncontrolled input </h2>
      <p>
        This input's state is <i> not </i> controlled by React, it relies on
        native DOM form submissions. Submitting will parse the form values.
      </p>
      <form onSubmit={submitForm}>
        <label>
          Name:
          <StyledInput placeholder="Type in here" name="myInput" type="text" />
        </label>
        <input type="submit" value="Submit" />
      </form>
      <table className="table mt-3">
        <thead className="thead-dark">
          <tr>
            <th>First Name</th>
            <th>Last Name</th>
            <th>Email</th>
            <th>Avatar</th>
          </tr>
        </thead>
        <tbody>
          {userList.map((x, i) => (
            <tr key={i}>
              <td>{x.first_name}</td>
              <td>{x.last_name}</td>
              <td>{x.email}</td>
              <td>
                <img src={x.avatar} width="50" height="50" alt={x.avatar} />
              </td>
            </tr>
          ))}
          {userList.length === 0 && (
            <tr>
              <td className="text-center" colSpan="4">
                <b>No data found to display.</b>
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
