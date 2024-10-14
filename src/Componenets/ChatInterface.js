// // src/components/ChatInterface.js

// import React, { useState, useRef, useEffect } from 'react';
// import axios from 'axios';

// const ChatInterface = ({ treeCount, totalArea }) => {
//   const [query, setQuery] = useState('');
//   const [conversation, setConversation] = useState([]);
//   const messagesEndRef = useRef(null);

//   const handleSend = async () => {
//     if (!query.trim()) return;

//     const userMessage = { sender: 'user', text: query };
//     setConversation((prev) => [...prev, userMessage]);

//     try {
//       const response = await axios.post('http://localhost:8000/chat/', {
//         query,
//         tree_count: treeCount,
//         total_area: totalArea,
//       });
//       const assistantMessage = { sender: 'assistant', text: response.data.response };
//       setConversation((prev) => [...prev, assistantMessage]);
//     } catch (error) {
//       console.error('Error in chat:', error);
//       const errorMessage = { sender: 'assistant', text: 'An error occurred during chat processing.' };
//       setConversation((prev) => [...prev, errorMessage]);
//     }

//     setQuery('');
//   };

//   // Auto-scroll to the bottom when a new message is added
//   useEffect(() => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   }, [conversation]);

//   return (
//     <div className="p-4 bg-white shadow rounded h-full flex flex-col">
//       <h2 className="text-2xl font-bold mb-4">Chat Interface</h2>
//       <div className="flex-grow overflow-y-auto mb-4">
//         {conversation.map((msg, index) => (
//           <div
//             key={index}
//             className={`mb-2 ${msg.sender === 'user' ? 'text-right' : 'text-left'}`}
//           >
//             <span
//               className={`inline-block px-4 py-2 rounded ${
//                 msg.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-300 text-black'
//               }`}
//             >
//               {msg.text}
//             </span>
//           </div>
//         ))}
//         <div ref={messagesEndRef} />
//       </div>
//       <div className="flex">
//         <input
//           type="text"
//           value={query}
//           onChange={(e) => setQuery(e.target.value)}
//           className="flex-grow border rounded-l px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
//           placeholder="Type your message..."
//         />
//         <button
//           onClick={handleSend}
//           className="px-4 py-2 bg-blue-500 text-white rounded-r hover:bg-blue-600"
//         >
//           Send
//         </button>
//       </div>
//     </div>
//   );
// };

// export default ChatInterface;

import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const ChatInterface = ({ treeCount, totalArea, averageTreeArea, imageMetadata }) => {
  const [query, setQuery] = useState('');
  const [conversation, setConversation] = useState([]);
  const messagesEndRef = useRef(null);

  const handleSend = async () => {
    if (!query.trim()) return;

    const userMessage = { sender: 'user', text: query };
    setConversation((prev) => [...prev, userMessage]);

    try {
      const response = await axios.post('http://localhost:8000/chat/', {
        query,
        tree_count: treeCount,
        total_area: totalArea,
        average_tree_area: averageTreeArea,
        image_metadata: imageMetadata,
      });
      const assistantMessage = { sender: 'assistant', text: response.data.response };
      setConversation((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error in chat:', error);
      const errorMessage = { sender: 'assistant', text: 'An error occurred during chat processing.' };
      setConversation((prev) => [...prev, errorMessage]);
    }

    setQuery('');
  };

  // Auto-scroll to the bottom when a new message is added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation]);

  return (
    <div className="p-4 bg-white shadow rounded h-full flex flex-col">
      <h2 className="text-2xl font-bold mb-4">Chat Interface</h2>
      <div className="flex-grow overflow-y-auto mb-4">
        {conversation.map((msg, index) => (
          <div
            key={index}
            className={`mb-2 ${msg.sender === 'user' ? 'text-right' : 'text-left'}`}
          >
            <span
              className={`inline-block px-4 py-2 rounded ${
                msg.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-300 text-black'
              }`}
            >
              {msg.text}
            </span>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="flex">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-grow border rounded-l px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Type your message..."
        />
        <button
          onClick={handleSend}
          className="px-4 py-2 bg-blue-500 text-white rounded-r hover:bg-blue-600"
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;
