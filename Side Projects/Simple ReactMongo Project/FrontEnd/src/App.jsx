import React, { useState } from 'react';
import UserForm from './UserForm';
import UserList from './UserList';

const App = () => {
  const [users, setUsers] = useState([]);

  const handleUserAdded = (newUser) => {
    setUsers((prevUsers) => [...prevUsers, newUser]);
  };

  return (
    <div className="App">
      <h1>User Management</h1>
      <UserForm onUserAdded={handleUserAdded} />
      <UserList users={users} />
    </div>
  );
};

export default App;
