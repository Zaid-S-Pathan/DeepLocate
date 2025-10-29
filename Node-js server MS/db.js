const mongoose=require('mongoose');
const mongoURI="mongodb+srv://zaidspathan:jsGQt5SejcyBBSid@cluster0.mko8h.mongodb.net/deepLocate?retryWrites=true&w=majority&appName=Cluster0"

const DbConnect=()=>{
    mongoose.connect(mongoURI,()=>{
        console.log("connected to mongo successfully");
    })
}
module.exports=DbConnect;




