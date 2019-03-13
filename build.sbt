name := "WebDeepLearning"

version := "1.0"

lazy val `webdeeplearning` = (project in file(".")).enablePlugins(PlayScala, LauncherJarPlugin)

scalaVersion := "2.11.11"

libraryDependencies ++= Seq( jdbc , cache , ws   , specs2 % Test )

unmanagedResourceDirectories in Test <+=  baseDirectory ( _ /"target/web/public/test" )

unmanagedBase := baseDirectory.value / "lib"

libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13",
  "org.scalanlp" %% "breeze-natives" % "0.13",
  "org.scalanlp" %% "breeze-viz" % "0.13",
  "net.liftweb" % "lift-json_2.11" % "3.0.1",
  "org.apache.httpcomponents" % "httpclient" % "4.5.7"
)

libraryDependencies ++= Seq(
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta3",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-beta3",
  "org.deeplearning4j" % "deeplearning4j-ui_2.11" % "1.0.0-beta3"
)

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta3"

libraryDependencies ++= Seq(
  "com.twelvemonkeys.imageio" % "imageio" % "3.1.2",
  "com.twelvemonkeys.imageio" % "imageio-core" % "3.1.2",
  "com.twelvemonkeys.common" % "common-lang" % "3.1.2"
)


resolvers += "scalaz-bintray" at "https://dl.bintray.com/scalaz/releases"
