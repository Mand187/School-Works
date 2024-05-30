import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import 'bootstrap/dist/css/bootstrap.min.css'
import Signup  from './pages/signUpPage'
import Login   from './pages/userLogPage'
import Landing from './pages/landingPage'

function App() {
  const [count, setCount] = useState(0)

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