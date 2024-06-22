import { useState } from 'react'
<<<<<<< HEAD
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
=======
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import 'bootstrap/dist/css/bootstrap.min.css'
import Signup  from './pages/signUpPage'
import Login   from './pages/userLogPage'
import Landing from './pages/landingPage'
>>>>>>> 937e2735292f1083df7a56052ad647657feef5d8

function App() {
  const [count, setCount] = useState(0)

<<<<<<< HEAD
  return (
    <>
      <div>
        <a href="https://vitejs.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.jsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
}

export default App
=======
    return (
      <BrowserRouter>
      <Routes>
        <Route path='/'         element={<Landing />}></Route>
        <Route path='/register' element={<Signup />}></Route>
        <Route path='/login'    element={<Login />}></Route>
      </Routes>
      </BrowserRouter>
    )
}

export default App
>>>>>>> 937e2735292f1083df7a56052ad647657feef5d8
