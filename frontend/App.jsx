import "./styles.css";
import React, { useState } from "react";
import styled from "styled-components";
import axios from "axios";

const Navbar = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 60px;
  background-color: #0f172a; /* dark blue-gray */
  color: white;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  z-index: 1000;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h1`
  font-size: 20px;
  margin: 0;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 10px;
`;

const NavButton = styled.a`
  background-color: #1e293b;
  color: white;
  padding: 8px 14px;
  text-decoration: none;
  border-radius: 6px;
  font-size: 14px;
  transition: background 0.2s ease;

  &:hover {
    background-color: #334155;
  }
`;

function TopNavbar() {
  return (
    <Navbar>
      <Title>📈 Stock Prediction & MLOps Manger</Title>
      <ButtonGroup>
        <NavButton href="http://localhost:8080" target="_blank">
          Airflow
        </NavButton>
        <NavButton href="http://localhost:5050" target="_blank">
          MLflow
        </NavButton>
      </ButtonGroup>
    </Navbar>
  );
}


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
  const inputProps = useInput();
  const [stockList, setStockList] = useState([]);
  const [loading, setLoading] = useState(false);

  async function submitForm(event) {
    event.preventDefault();
    const ticker = event.target.myInput.value.trim().toUpperCase();
    if (!ticker) return;

    setLoading(true);

    try {
      const res = await axios.get(`http://localhost:8000/predict`, {
        params: { ticker: ticker },
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
          "Content-Type": "application/json"
        }
      });

      const { ticker: tick, predicted_price, image_url } = res.data;

      setStockList((prev) => [
        {
          ticker: tick,
          predicted_price,
          change: null, // Optional: you can compute this if you want
          image_url,
        },
        ...prev,
      ]);
    } catch (err) {
      console.error("Prediction error:", err);
      alert("Failed to fetch prediction.");
    }

    setLoading(false);
  }

  return (
    <div>
      <TopNavbar />
      <form onSubmit={submitForm} style={{ paddingTop: "80px" }}>
        <label>
          Ticker:
          <StyledInput
            placeholder="e.g. AAPL, TSLA"
            name="myInput"
            type="text"
            {...inputProps}
          />
        </label>
        <input type="submit" value="Submit" disabled={loading} />
      </form>

      <table className="table mt-3">
        <thead className="thead-dark">
          <tr>
            <th>Tick</th>
            <th>Predicted Price</th>
            <th>Change</th>
            <th>Plot</th>
          </tr>
        </thead>
        <tbody>
          {stockList.map((x, i) => (
            <tr key={i}>
              <td>{x.ticker}</td>
              <td>{x.predicted_price.toFixed(2)}</td>
              <td>{x.change ?? "-"}</td>
              <td>
                <img
                  src={`http://localhost:8000${x.image_url}`}
                  width="80"
                  height="50"
                  alt={`Plot for ${x.ticker}`}
                />
              </td>
            </tr>
          ))}
          {stockList.length === 0 && (
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
